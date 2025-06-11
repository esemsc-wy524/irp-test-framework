"""
Modified trainer for KoopmanAE model
Adapted from the original trainer to work with the new model architecture and loss functions
"""

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import os
import json
import numpy as np
from typing import Dict, Optional, Tuple, List

# Import your modules
from models.koopman_ae import KoopmanAE
from utils.lossfn import KoopmanAELoss


def prepare_batch_data(batch: Dict, device: torch.device, config: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare batch data for training
    
    Returns:
        inputs: Dictionary of input tensors
        targets: Dictionary of target tensors
        metadata: Dictionary of physical parameters
    """
    # Get input sequences
    input_seq = batch['input'].to(device)  # [batch, time, space]
    
    # Flatten spatial dimensions if needed
    if input_seq.dim() == 4:  # 2D data: [batch, time, x, y]
        batch_size, time_steps, nx, ny = input_seq.shape
        input_seq = input_seq.reshape(batch_size, time_steps, -1)
    
    # For single time step, take the last frame
    if config.get('single_step_input', True):
        inputs = input_seq[:, -1, :]  # [batch, space]
    else:
        inputs = input_seq  # [batch, time, space]
    
    # Prepare targets
    targets = {
        'reconstruction': inputs  # Autoencoder reconstruction
    }
    
    # Add future targets if available
    if 'target' in batch:
        future_seq = batch['target'].to(device)
        if future_seq.dim() == 4:  # 2D data
            batch_size, time_steps, nx, ny = future_seq.shape
            future_seq = future_seq.reshape(batch_size, time_steps, -1)
        targets['future'] = future_seq
    
    # Add masks if available (for sparse observations)
    if 'mask' in batch:
        targets['mask'] = batch['mask'].to(device)
    
    # Prepare metadata for physics constraints
    metadata = {
        'dataset': config.get('dataset', 'heat1d'),  # Add dataset identifier
        'dx': config.get('dx', 0.01),
        'dy': config.get('dy', 0.01),
        'dt': config.get('dt', 0.01),
        'alpha': config.get('alpha', 0.01),  # Heat equation
        'D': config.get('D', 0.01),  # Advection-diffusion
        'vx': config.get('vx', 0.5),
        'vy': config.get('vy', 0.3),
        'nx': config.get('nx', 100),  # Grid dimensions
        'ny': config.get('ny', 64)
    }
    
    return inputs, targets, metadata


def train_model(
    model: KoopmanAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    optimizer: Optional[Dict] = None,
    lr_scheduler: Optional[Dict] = None,
    device: str = "cpu",
    save_path: Optional[str] = None,
    verbose: bool = True,
    loss_fn: Optional[KoopmanAELoss] = None,
    config: Optional[Dict] = None
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Train KoopmanAE model
    
    Args:
        model: KoopmanAE model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        optimizer: Optimizer configuration
        lr_scheduler: Learning rate scheduler configuration
        device: Device to train on
        save_path: Path to save best model
        verbose: Whether to print progress
        loss_fn: Loss function instance
        config: Additional configuration
        
    Returns:
        train_losses: Dictionary of training losses
        val_losses: Dictionary of validation losses
    """
    # Default config
    if config is None:
        config = {}
    
    # Move model to device
    device = torch.device(device)
    model.to(device)
    
    # Initialize optimizer
    if optimizer is None or optimizer['name'] == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer['name'] == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer['name'] == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer}")
    
    # Initialize learning rate scheduler
    if lr_scheduler is None:
        scheduler = StepLR(opt, step_size=50, gamma=0.97)
    elif lr_scheduler['name'] == 'StepLR':
        scheduler = StepLR(opt, step_size=lr_scheduler['step_size'], gamma=lr_scheduler['gamma'])
    elif lr_scheduler['name'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    elif lr_scheduler['name'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    else:
        raise ValueError(f"Unsupported learning rate scheduler type: {lr_scheduler}")
    
    # Initialize loss tracking
    train_losses = {
        'total': [],
        'recon': [],
        'pred': [],
        'physics': [],
        'linearity': []
    }
    
    val_losses = {
        'total': [],
        'recon': [],
        'pred': [],
        'physics': [],
        'linearity': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    epoch_pbar = tqdm(range(epochs), desc='Training', leave=True)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss_accum = {key: 0.0 for key in train_losses.keys()}
        train_samples = 0
        
        # Batch loop
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for batch_idx, batch in enumerate(batch_pbar):
            # Prepare data
            inputs, targets, metadata = prepare_batch_data(batch, device, config)
            
            # Forward pass
            n_pred_steps = targets['future'].shape[1] if 'future' in targets else 1
            outputs = model(inputs, n_steps=n_pred_steps, return_latent=True)
            
            # Compute losses
            losses = loss_fn(outputs, targets, model, metadata)
            
            # Backward pass
            opt.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            if config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            opt.step()
            
            # Accumulate losses
            batch_size = inputs.shape[0]
            for key in train_loss_accum:
                if key in losses:
                    train_loss_accum[key] += losses[key].item() * batch_size
            train_samples += batch_size
            
            # Update progress bar
            batch_pbar.set_postfix({
                'loss': f"{losses['total'].item():.6f}",
                'recon': f"{losses.get('recon', 0).item():.4f}",
                'pred': f"{losses.get('pred', 0).item():.4f}"
            })
        
        # Average training losses
        for key in train_losses:
            if train_samples > 0:
                avg_loss = train_loss_accum[key] / train_samples
                train_losses[key].append(avg_loss)
            else:
                train_losses[key].append(0.0)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss_accum = {key: 0.0 for key in val_losses.keys()}
            val_samples = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc='Validation', leave=False)
                
                for batch in val_pbar:
                    # Prepare data
                    inputs, targets, metadata = prepare_batch_data(batch, device, config)
                    
                    # Forward pass
                    n_pred_steps = targets['future'].shape[1] if 'future' in targets else 1
                    outputs = model(inputs, n_steps=n_pred_steps, return_latent=True)
                    
                    # Compute losses
                    losses = loss_fn(outputs, targets, model, metadata)
                    
                    # Accumulate losses
                    batch_size = inputs.shape[0]
                    for key in val_loss_accum:
                        if key in losses:
                            val_loss_accum[key] += losses[key].item() * batch_size
                    val_samples += batch_size
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f"{losses['total'].item():.6f}"
                    })
            
            # Average validation losses
            for key in val_losses:
                if val_samples > 0:
                    avg_loss = val_loss_accum[key] / val_samples
                    val_losses[key].append(avg_loss)
                else:
                    val_losses[key].append(0.0)
            
            current_val_loss = val_losses['total'][-1]
            
            # Learning rate scheduling
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_val_loss)
            else:
                scheduler.step()
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f"{train_losses['total'][-1]:.6f}",
                'val_loss': f"{current_val_loss:.6f}",
                'val_recon': f"{val_losses['recon'][-1]:.4f}",
                'val_pred': f"{val_losses['pred'][-1]:.4f}",
                'val_phys': f"{val_losses['physics'][-1]:.4f}",
                'lr': f"{opt.param_groups[0]['lr']:.2e}"
            })
            
            # Save best model
            if current_val_loss < best_val_loss and save_path:
                best_val_loss = current_val_loss
                best_epoch = epoch + 1
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                
                torch.save(checkpoint, save_path)
                
                # Save losses separately for easy plotting
                losses_path = save_path.replace('.pt', '_losses.json')
                with open(losses_path, 'w') as f:
                    json.dump({
                        'train_losses': {k: [float(v) for v in vals] for k, vals in train_losses.items()},
                        'val_losses': {k: [float(v) for v in vals] for k, vals in val_losses.items()},
                        'best_epoch': best_epoch,
                        'best_val_loss': float(best_val_loss)
                    }, f, indent=2)
                
                if verbose:
                    epoch_pbar.write(
                        f'Epoch {epoch+1}: New best model saved! '
                        f'(val_loss: {current_val_loss:.6f}, '
                        f'recon: {val_losses["recon"][-1]:.4f}, '
                        f'pred: {val_losses["pred"][-1]:.4f})'
                    )
        else:
            # No validation, just step scheduler
            if not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
    
    if save_path and verbose:
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        print(f"Best model saved to: {save_path}")
    
    return train_losses, val_losses


def load_checkpoint(checkpoint_path: str, model: KoopmanAE, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cpu') -> Dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load model to
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def evaluate_model(model: KoopmanAE, test_loader: DataLoader, 
                  loss_fn: KoopmanAELoss, device: str = 'cpu',
                  config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run on
        config: Configuration dictionary
        
    Returns:
        Dictionary of test metrics
    """
    if config is None:
        config = {}
    
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    test_losses = {
        'total': 0.0,
        'recon': 0.0,
        'pred': 0.0,
        'physics': 0.0,
        'linearity': 0.0
    }
    
    test_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # Prepare data
            inputs, targets, metadata = prepare_batch_data(batch, device, config)
            
            # Forward pass
            n_pred_steps = targets['future'].shape[1] if 'future' in targets else 1
            outputs = model(inputs, n_steps=n_pred_steps, return_latent=True)
            
            # Compute losses
            losses = loss_fn(outputs, targets, model, metadata)
            
            # Accumulate losses
            batch_size = inputs.shape[0]
            for key in test_losses:
                if key in losses:
                    test_losses[key] += losses[key].item() * batch_size
            test_samples += batch_size
    
    # Average losses
    for key in test_losses:
        if test_samples > 0:
            test_losses[key] /= test_samples
    
    return test_losses