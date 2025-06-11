"""
Main training script for KoopmanAE
This script shows how to use the modified trainer with the new model
"""

import torch
import yaml
import argparse
import os
from datetime import datetime

# Import your modules
from models.koopman_ae import create_koopman_ae
from utils.lossfn import create_loss_function
from utils.trainer import train_model, load_checkpoint, evaluate_model
from utils.pde_dataloader import create_dataloaders, create_dataloaders_debug


def main(args):
    """Main training function"""
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.dataset:
        config['dataset'] = args.dataset
    if args.epochs:
        config['n_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Set data path based on dataset
    if config['dataset'] == 'heat1d':
        config['data_path'] = 'data/1DHeatEq/heat1d_data.h5'
        config['nx'] = 100  # Number of spatial points
        # Physics parameters for heat equation
        config['dx'] = 0.01
        config['dt'] = 0.001
        config['alpha'] = 0.01
    elif config['dataset'] == 'advection_diffusion_2d':
        config['data_path'] = 'data/2DAdvdiffEq/advection_diffusion_2d_data.h5'
        config['nx'] = 64
        config['ny'] = 64
        # Physics parameters for advection-diffusion
        config['dx'] = 0.02
        config['dy'] = 0.02
        config['dt'] = 0.001
        config['D'] = 0.01
        config['vx'] = 0.5
        config['vy'] = 0.3
    
    # Create data loaders
    print(f"\nLoading dataset: {config['dataset']}")
    if args.debug:
        train_loader, val_loader, test_loader = create_dataloaders_debug(
            config['data_path'],
            batch_size=config.get('batch_size', 32),
            input_steps=config.get('input_steps', 10),
            output_steps=config.get('output_steps', 10),
            num_workers=config.get('num_workers', 4),
            normalize=config.get('normalize', True),
            stride=config.get('stride', 1),
            debug_mode=True,
            debug_samples=args.debug_samples
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            config['data_path'],
            batch_size=config.get('batch_size', 32),
            input_steps=config.get('input_steps', 10),
            output_steps=config.get('output_steps', 10),
            num_workers=config.get('num_workers', 4),
            normalize=config.get('normalize', True),
            stride=config.get('stride', 1)
        )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = create_koopman_ae(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = create_loss_function(config)
    
    # Set up optimizer configuration
    optimizer_config = config.get('optimizer', {'name': 'Adam'})
    scheduler_config = config.get('lr_scheduler', {'name': 'CosineAnnealingLR'})
    
    # Create save path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"koopman_ae_{config['dataset']}_{timestamp}"
    save_dir = os.path.join('results', 'checkpoints', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pt')
    
    # Save configuration
    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nModel and results will be saved to: {save_dir}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('n_epochs', 100),
        lr=config.get('learning_rate', 1e-3),
        optimizer=optimizer_config,
        lr_scheduler=scheduler_config,
        device=device,
        save_path=save_path,
        verbose=True,
        loss_fn=loss_fn,
        config=config
    )
    
    # Plot training curves
    if not args.no_plot:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Curves - {config["dataset"]}')
        
        # Total loss
        axes[0, 0].plot(train_losses['total'], label='Train')
        axes[0, 0].plot(val_losses['total'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(train_losses['recon'], label='Train')
        axes[0, 1].plot(val_losses['recon'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Prediction loss
        axes[1, 0].plot(train_losses['pred'], label='Train')
        axes[1, 0].plot(val_losses['pred'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Prediction Loss')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Physics loss
        axes[1, 1].plot(train_losses['physics'], label='Train')
        axes[1, 1].plot(val_losses['physics'], label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Physics Constraint Loss')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to: {plot_path}")
    
    # Evaluate on test set
    # if not args.skip_test:
    #     print("\nEvaluating on test set...")
        
    #     # Load best model
    #     checkpoint = load_checkpoint(save_path, model, device=device)
    #     print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
    #     # Evaluate
    #     test_losses = evaluate_model(model, test_loader, loss_fn, device, config)
        
    #     print("\nTest Set Results:")
    #     print(f"  Total Loss: {test_losses['total']:.6f}")
    #     print(f"  Reconstruction Loss: {test_losses['recon']:.6f}")
    #     print(f"  Prediction Loss: {test_losses['pred']:.6f}")
    #     print(f"  Physics Loss: {test_losses['physics']:.6f}")
    #     print(f"  Linearity Loss: {test_losses['linearity']:.6f}")
        
    #     # Save test results
    #     import json
    #     test_results_path = os.path.join(save_dir, 'test_results.json')
    #     with open(test_results_path, 'w') as f:
    #         json.dump({k: float(v) for k, v in test_losses.items()}, f, indent=2)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train KoopmanAE model')
    parser.add_argument('--config', type=str, default='configs/KoopmanAE.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['heat1d', 'advection_diffusion_2d'],
                        help='Dataset to use (overrides config)')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate (overrides config)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting training curves')
    # parser.add_argument('--skip_test', action='store_true',
    #                     help='Skip test set evaluation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with limited samples')
    parser.add_argument('--debug_samples', type=int, default=100,
                        help='Number of samples to use in debug mode (default: 100)')
    
    args = parser.parse_args()
    main(args)