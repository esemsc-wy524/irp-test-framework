"""
Loss functions for Koopman Autoencoder with physics constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Tuple
import numpy as np


class ReconstructionLoss(nn.Module):
    """Reconstruction loss between input and decoded output"""
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'huber':
            self.loss_fn = nn.HuberLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask for sparse observations
        """
        if mask is not None:
            # Apply mask for sparse observations
            pred_masked = pred[mask]
            target_masked = target[mask]
            return self.loss_fn(pred_masked, target_masked)
        else:
            return self.loss_fn(pred, target)


class PredictionLoss(nn.Module):
    """Multi-step prediction loss"""
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean',
                 time_weight_decay: float = 1.0):
        super().__init__()
        self.base_loss = ReconstructionLoss(loss_type, reduction)
        self.time_weight_decay = time_weight_decay
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: [batch, n_steps, ...]
            targets: [batch, n_steps, ...]
            mask: Optional mask for sparse observations
        """
        n_steps = predictions.shape[1]
        total_loss = 0.0
        
        for t in range(n_steps):
            # Apply time decay weight
            weight = self.time_weight_decay ** t
            step_loss = self.base_loss(predictions[:, t], targets[:, t], 
                                      mask[:, t] if mask is not None else None)
            total_loss += weight * step_loss
        
        return total_loss / n_steps


class LinearityLoss(nn.Module):
    """Enforce linearity of dynamics in latent space"""
    
    def __init__(self, n_samples: int = 100):
        super().__init__()
        self.n_samples = n_samples
    
    def forward(self, koopman_op: nn.Module, latent_dim: int, 
                device: torch.device) -> torch.Tensor:
        """
        Test linearity: K(αz1 + βz2) = αK(z1) + βK(z2)
        """
        # Sample random latent vectors
        z1 = torch.randn(self.n_samples, latent_dim, device=device)
        z2 = torch.randn(self.n_samples, latent_dim, device=device)
        
        # Random coefficients
        alpha = torch.rand(self.n_samples, 1, device=device)
        beta = torch.rand(self.n_samples, 1, device=device)
        
        # Test linearity
        z_combined = alpha * z1 + beta * z2
        Kz_combined = koopman_op(z_combined)
        Kz_linear = alpha * koopman_op(z1) + beta * koopman_op(z2)
        
        linearity_error = F.mse_loss(Kz_combined, Kz_linear)
        
        return linearity_error


class PhysicsConstraintLoss(nn.Module):
    """
    Physics constraint loss F(H(z)) ≈ 0
    Phase 1: Simple L2 penalty
    """
    
    def __init__(self, constraint_type: str = 'heat_equation'):
        super().__init__()
        self.constraint_type = constraint_type
    
    def forward(self, y: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Compute physics constraint violation
        
        Args:
            y: Physical space output [batch, time, space] or [batch, time, x, y]
            metadata: Dictionary containing physical parameters
        """
        if self.constraint_type == 'heat_equation':
            return self._heat_equation_constraint(y, metadata)
        elif self.constraint_type == 'advection_diffusion':
            return self._advection_diffusion_constraint(y, metadata)
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
    
    def _heat_equation_constraint(self, u: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Heat equation constraint: ∂u/∂t - α∇²u = 0
        Using finite differences
        """
        alpha = metadata.get('alpha', 0.01)
        dx = metadata.get('dx', 0.01)
        dt = metadata.get('dt', 0.01)
        
        # Handle different input dimensions
        if u.dim() == 2:  # [batch, space] - single time step
            # For single time step, we can only compute spatial constraint
            # Return spatial smoothness regularization instead
            batch_size, space_dim = u.shape
            
            # Spatial second derivative (central difference)
            d2u_dx2 = torch.zeros_like(u)
            d2u_dx2[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
            
            # Boundary conditions (Neumann: du/dx = 0)
            d2u_dx2[:, 0] = (u[:, 1] - u[:, 0]) / (dx**2)
            d2u_dx2[:, -1] = (u[:, -2] - u[:, -1]) / (dx**2)
            
            # Return smoothness regularization (penalize large second derivatives)
            return torch.mean(d2u_dx2**2) * 0.1  # Scale down since we can't compute time derivative
            
        elif u.dim() == 3:  # [batch, time, space] - multiple time steps
            # Time derivative (forward difference)
            if u.shape[1] < 2:
                # Not enough time steps for time derivative
                return torch.tensor(0.0, device=u.device)
            
            du_dt = (u[:, 1:, :] - u[:, :-1, :]) / dt
            
            # Spatial second derivative (central difference)
            d2u_dx2 = torch.zeros_like(u[:, :-1, :])
            d2u_dx2[:, :, 1:-1] = (u[:, :-1, 2:] - 2*u[:, :-1, 1:-1] + u[:, :-1, :-2]) / (dx**2)
            
            # Boundary conditions (Neumann: du/dx = 0)
            d2u_dx2[:, :, 0] = (u[:, :-1, 1] - u[:, :-1, 0]) / (dx**2)
            d2u_dx2[:, :, -1] = (u[:, :-1, -2] - u[:, :-1, -1]) / (dx**2)
            
            # Physics residual
            residual = du_dt - alpha * d2u_dx2
            
            # Return mean squared residual
            return torch.mean(residual**2)
        
        else:
            raise ValueError(f"Unexpected tensor dimension: {u.dim()}")
    
    def _advection_diffusion_constraint(self, u: torch.Tensor, 
                                       metadata: Dict) -> torch.Tensor:
        """
        Advection-diffusion constraint: ∂u/∂t + v·∇u - D∇²u = 0
        """
        D = metadata.get('D', 0.01)
        vx = metadata.get('vx', 0.5)
        vy = metadata.get('vy', 0.3)
        dx = metadata.get('dx', 0.02)
        dy = metadata.get('dy', 0.02)
        dt = metadata.get('dt', 0.01)
        
        # Handle different input dimensions
        if u.dim() == 2:  # [batch, space] - flattened single time step
            # For flattened 2D data, return smoothness regularization
            return torch.tensor(0.0, device=u.device)  # Skip for now
            
        elif u.dim() == 3:  # [batch, x*y] or [batch, time, x*y] 
            if u.shape[1] < 10:  # Likely [batch, time, flattened_space]
                # Not enough time steps or flattened spatial data
                return torch.tensor(0.0, device=u.device)
            else:
                # Might be [batch, x, y] single time step
                # Return smoothness regularization
                batch_size, nx, ny = u.shape
                
                # Compute spatial derivatives for smoothness
                d2u_dx2 = torch.zeros_like(u)
                d2u_dy2 = torch.zeros_like(u)
                
                d2u_dx2[:, 1:-1, :] = (u[:, 2:, :] - 2*u[:, 1:-1, :] + u[:, :-2, :]) / (dx**2)
                d2u_dy2[:, :, 1:-1] = (u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]) / (dy**2)
                
                laplacian = d2u_dx2 + d2u_dy2
                return torch.mean(laplacian**2) * 0.1
                
        elif u.dim() == 4:  # [batch, time, x, y] - proper format
            # Time derivative
            if u.shape[1] < 2:
                return torch.tensor(0.0, device=u.device)
                
            du_dt = (u[:, 1:, :, :] - u[:, :-1, :, :]) / dt
            
            # Spatial derivatives (central differences)
            # First derivatives for advection
            du_dx = torch.zeros_like(u[:, :-1, :, :])
            du_dy = torch.zeros_like(u[:, :-1, :, :])
            
            du_dx[:, :, 1:-1, :] = (u[:, :-1, 2:, :] - u[:, :-1, :-2, :]) / (2*dx)
            du_dy[:, :, :, 1:-1] = (u[:, :-1, :, 2:] - u[:, :-1, :, :-2]) / (2*dy)
            
            # Second derivatives for diffusion
            d2u_dx2 = torch.zeros_like(u[:, :-1, :, :])
            d2u_dy2 = torch.zeros_like(u[:, :-1, :, :])
            
            d2u_dx2[:, :, 1:-1, :] = (u[:, :-1, 2:, :] - 2*u[:, :-1, 1:-1, :] + 
                                      u[:, :-1, :-2, :]) / (dx**2)
            d2u_dy2[:, :, :, 1:-1] = (u[:, :-1, :, 2:] - 2*u[:, :-1, :, 1:-1] + 
                                      u[:, :-1, :, :-2]) / (dy**2)
            
            # Physics residual
            residual = du_dt + vx * du_dx + vy * du_dy - D * (d2u_dx2 + d2u_dy2)
            
            return torch.mean(residual**2)
        
        else:
            raise ValueError(f"Unexpected tensor dimension: {u.dim()}")


class KoopmanAELoss(nn.Module):
    """
    Combined loss function for Koopman Autoencoder
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        pred_weight: float = 1.0,
        physics_weight: float = 0.1,
        linearity_weight: float = 0.01,
        loss_type: str = 'mse',
        constraint_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.recon_weight = recon_weight
        self.pred_weight = pred_weight
        self.physics_weight = physics_weight
        self.linearity_weight = linearity_weight
        
        # Initialize loss components
        self.recon_loss = ReconstructionLoss(loss_type)
        self.pred_loss = PredictionLoss(loss_type, time_weight_decay=kwargs.get('time_decay', 1.0))
        self.linearity_loss = LinearityLoss()
        
        if constraint_type is not None:
            self.physics_loss = PhysicsConstraintLoss(constraint_type)
        else:
            self.physics_loss = None
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: nn.Module,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components
        
        Args:
            model_output: Dictionary with 'reconstruction', 'predictions', etc.
            targets: Dictionary with 'reconstruction', 'future', etc.
            model: The KoopmanAE model (for accessing Koopman operator)
            metadata: Physical parameters for constraint computation
        
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Reconstruction loss
        if 'reconstruction' in model_output and 'reconstruction' in targets:
            losses['recon'] = self.recon_loss(
                model_output['reconstruction'],
                targets['reconstruction'],
                targets.get('mask', None)
            )
        
        # Prediction loss
        if 'predictions' in model_output and 'future' in targets:
            losses['pred'] = self.pred_loss(
                model_output['predictions'],
                targets['future'],
                targets.get('future_mask', None)
            )
        
        # Linearity loss (regularization)
        if self.linearity_weight > 0:
            device = next(model.parameters()).device
            losses['linearity'] = self.linearity_loss(
                model.koopman,
                model.latent_dim,
                device
            )
        
        # Physics constraint loss
        if self.physics_loss is not None and self.physics_weight > 0:
            # Try to use predictions for physics constraints if available
            if 'predictions' in model_output and model_output['predictions'] is not None:
                # Concatenate reconstruction and predictions for time sequence
                recon = model_output['reconstruction']
                pred = model_output['predictions']
                
                # Reshape if needed
                if recon.dim() == 2:  # [batch, space]
                    recon = recon.unsqueeze(1)  # [batch, 1, space]
                
                # Create time sequence
                if pred.dim() == 3:  # [batch, time, space]
                    time_sequence = torch.cat([recon, pred], dim=1)  # [batch, time+1, space]
                else:
                    time_sequence = recon
                
                # For 2D data, might need to reshape
                if metadata.get('dataset') == 'advection_diffusion_2d' and 'nx' in metadata:
                    nx = metadata.get('nx', 64)
                    ny = metadata.get('ny', 64)
                    if time_sequence.shape[-1] == nx * ny:
                        batch, time, flat_space = time_sequence.shape
                        time_sequence = time_sequence.reshape(batch, time, nx, ny)
                
                losses['physics'] = self.physics_loss(time_sequence, metadata)
                
            elif 'reconstruction' in model_output and metadata is not None:
                # Use only reconstruction (will compute spatial regularization)
                losses['physics'] = self.physics_loss(
                    model_output['reconstruction'],
                    metadata
                )
        
        # Compute total loss
        total_loss = 0.0
        if 'recon' in losses:
            total_loss += self.recon_weight * losses['recon']
        if 'pred' in losses:
            total_loss += self.pred_weight * losses['pred']
        if 'linearity' in losses:
            total_loss += self.linearity_weight * losses['linearity']
        if 'physics' in losses:
            total_loss += self.physics_weight * losses['physics']
        
        losses['total'] = total_loss
        
        return losses


# Utility functions for specific equation types
def create_loss_function(config: Dict) -> KoopmanAELoss:
    """Create loss function from configuration"""
    loss_config = config.get('loss', {})
    
    # Determine constraint type based on dataset
    dataset = config.get('dataset', 'heat1d')
    if dataset == 'heat1d':
        constraint_type = 'heat_equation'
    elif dataset in ['advection_diffusion_2d', 'rotating_flow_2d']:
        constraint_type = 'advection_diffusion'
    else:
        constraint_type = None
    
    loss_fn = KoopmanAELoss(
        recon_weight=loss_config.get('recon_weight', 1.0),
        pred_weight=loss_config.get('pred_weight', 1.0),
        physics_weight=loss_config.get('physics_weight', 0.1),
        linearity_weight=loss_config.get('linearity_weight', 0.01),
        loss_type=loss_config.get('type', 'mse'),
        constraint_type=constraint_type,
        time_decay=loss_config.get('time_decay', 1.0)
    )
    
    return loss_fn


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop