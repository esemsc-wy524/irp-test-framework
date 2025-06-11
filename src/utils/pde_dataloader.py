"""
PDE Data Loader for Heat and Advection-Diffusion Equations
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import os
from typing import Dict, Optional, Tuple, List

class PDEDataset(Dataset):
    """PyTorch Dataset for PDE data"""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        input_steps: int = 10,
        output_steps: int = 10,
        stride: int = 1,
        normalize: bool = True,
        use_sparse: bool = False
    ):
        """
        Args:
            data_path: Path to HDF5 file
            split: 'train', 'val', or 'test'
            input_steps: Number of input time steps
            output_steps: Number of output time steps to predict
            stride: Stride for creating sequences
            normalize: Whether to normalize data
            use_sparse: Whether to use sparse observations (if available)
        """
        self.data_path = data_path
        self.split = split
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.stride = stride
        self.normalize = normalize
        self.use_sparse = use_sparse
        
        # Load data
        with h5py.File(data_path, 'r') as f:
            self.u = f[split]['u'][:]
            self.x = f[split]['x'][:]
            if 'y' in f[split]:
                self.y = f[split]['y'][:]
                self.is_2d = True
            else:
                self.is_2d = False
            self.t = f[split]['t'][:]
            
            # Load sparse observations if available
            if use_sparse and 'u_obs' in f[split]:
                self.u_obs = f[split]['u_obs'][:]
                self.mask = f[split]['mask'][:]
            else:
                self.u_obs = None
                self.mask = None
            
            # Load metadata
            self.metadata = dict(f.attrs)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        # Compute normalization statistics
        if self.normalize:
            self._compute_stats()
    
    def _create_sequences(self) -> List[Tuple[int, int]]:
        """Create sequence indices"""
        sequences = []
        n_samples, n_time = self.u.shape[:2]
        
        for i in range(n_samples):
            for t in range(0, n_time - self.input_steps - self.output_steps + 1, self.stride):
                sequences.append((i, t))
        
        return sequences
    
    def _compute_stats(self):
        """Compute mean and std for normalization"""
        # Use only training data for normalization
        if self.split == 'train':
            self.mean = np.mean(self.u)
            self.std = np.std(self.u)
            # Save stats
            stats_path = self.data_path.replace('.h5', '_stats.npz')
            np.savez(stats_path, mean=self.mean, std=self.std)
        else:
            # Load stats from training
            stats_path = self.data_path.replace('.h5', '_stats.npz')
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.mean = stats['mean']
                self.std = stats['std']
            else:
                print("Warning: No normalization stats found, computing from current split")
                self.mean = np.mean(self.u)
                self.std = np.std(self.u)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence"""
        sample_idx, time_idx = self.sequences[idx]
        
        # Extract input and output sequences
        input_seq = self.u[sample_idx, time_idx:time_idx+self.input_steps]
        output_seq = self.u[sample_idx, time_idx+self.input_steps:time_idx+self.input_steps+self.output_steps]
        
        # Normalize
        if self.normalize:
            input_seq = (input_seq - self.mean) / (self.std + 1e-8)
            output_seq = (output_seq - self.mean) / (self.std + 1e-8)
        
        # Convert to torch tensors
        data = {
            'input': torch.FloatTensor(input_seq),
            'target': torch.FloatTensor(output_seq),
            'sample_idx': sample_idx,
            'time_idx': time_idx
        }
        
        # Add sparse observations if available
        if self.use_sparse and self.u_obs is not None:
            input_obs = self.u_obs[sample_idx, time_idx:time_idx+self.input_steps]
            mask_seq = self.mask[time_idx:time_idx+self.input_steps]
            
            if self.normalize:
                # Only normalize non-NaN values
                valid_mask = ~np.isnan(input_obs)
                input_obs[valid_mask] = (input_obs[valid_mask] - self.mean) / (self.std + 1e-8)
            
            data['input_sparse'] = torch.FloatTensor(input_obs)
            data['mask'] = torch.BoolTensor(mask_seq)
        
        # Add coordinates
        data['x'] = torch.FloatTensor(self.x)
        if self.is_2d:
            data['y'] = torch.FloatTensor(self.y)
        
        return data
    
    def get_full_trajectory(self, idx: int) -> Dict[str, np.ndarray]:
        """Get full trajectory for visualization"""
        trajectory = {
            'u': self.u[idx],
            'x': self.x,
            't': self.t
        }
        
        if self.is_2d:
            trajectory['y'] = self.y
        
        if self.u_obs is not None:
            trajectory['u_obs'] = self.u_obs[idx]
            trajectory['mask'] = self.mask
        
        return trajectory


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    input_steps: int = 10,
    output_steps: int = 10,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create datasets
    train_dataset = PDEDataset(
        data_path, split='train', 
        input_steps=input_steps,
        output_steps=output_steps,
        **kwargs
    )
    
    val_dataset = PDEDataset(
        data_path, split='val',
        input_steps=input_steps,
        output_steps=output_steps,
        **kwargs
    )
    
    test_dataset = PDEDataset(
        data_path, split='test',
        input_steps=input_steps,
        output_steps=output_steps,
        use_sparse=True,  # Test set typically has sparse observations
        **kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_dataloaders_debug(
    data_path: str,
    batch_size: int = 32,
    input_steps: int = 10,
    output_steps: int = 10,
    num_workers: int = 4,
    debug_mode: bool = False,
    debug_samples: int = 100,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders with debug mode
    
    Args:
        data_path: Path to HDF5 file
        batch_size: Batch size
        input_steps: Number of input time steps
        output_steps: Number of output time steps
        num_workers: Number of data loading workers
        debug_mode: Whether to use debug mode with limited samples
        debug_samples: Number of samples to use in debug mode
        **kwargs: Additional arguments for PDEDataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create full datasets
    train_dataset = PDEDataset(
        data_path, split='train', 
        input_steps=input_steps,
        output_steps=output_steps,
        **kwargs
    )
    
    val_dataset = PDEDataset(
        data_path, split='val',
        input_steps=input_steps,
        output_steps=output_steps,
        **kwargs
    )
    
    test_dataset = PDEDataset(
        data_path, split='test',
        input_steps=input_steps,
        output_steps=output_steps,
        use_sparse=True,
        **kwargs
    )
    
    # Apply debug mode if requested
    if debug_mode:
        print(f"\n[DEBUG MODE] Using only {debug_samples} samples per split")
        
        # Limit training samples
        train_indices = list(range(min(debug_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        
        # Limit validation samples (use fewer for val)
        val_samples = max(debug_samples // 5, 10)
        val_indices = list(range(min(val_samples, len(val_dataset))))
        val_dataset = Subset(val_dataset, val_indices)
        
        # Limit test samples
        test_samples = max(debug_samples // 5, 10)
        test_indices = list(range(min(test_samples, len(test_dataset))))
        test_dataset = Subset(test_dataset, test_indices)
        
        # Reduce batch size in debug mode
        batch_size = min(batch_size, 16)
        print(f"[DEBUG MODE] Batch size reduced to: {batch_size}")
        
        # Use fewer workers in debug mode
        num_workers = min(num_workers, 2)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Only pin memory if GPU available
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


# Example usage and data inspection
def inspect_dataset(data_path: str):
    """Inspect dataset contents"""
    print(f"\nInspecting dataset: {data_path}")
    print("-" * 50)
    
    with h5py.File(data_path, 'r') as f:
        print("Groups:", list(f.keys()))
        
        for split in ['train', 'val', 'test']:
            if split in f:
                print(f"\n{split} split:")
                for key in f[split].keys():
                    shape = f[split][key].shape
                    print(f"  {key}: {shape}")
        
        print("\nMetadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
    
    # Test dataloader
    print("\nTesting dataloader...")
    dataset = PDEDataset(data_path, split='train', input_steps=5, output_steps=5)
    print(f"Number of sequences: {len(dataset)}")
    
    # Get one sample
    sample = dataset[0]
    print("\nSample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Example paths
    heat1d_path = "data/1DHeatEq/heat1d_data.h5"
    adv_diff_2d_path = "data/2DAdvdiffEq/advection_diffusion_2d_data.h5"
    
    # Inspect datasets if they exist
    if os.path.exists(heat1d_path):
        inspect_dataset(heat1d_path)
    
    if os.path.exists(adv_diff_2d_path):
        inspect_dataset(adv_diff_2d_path)
    
    # Create dataloaders
    if os.path.exists(heat1d_path):
        print("\nCreating dataloaders for Heat 1D...")
        train_loader, val_loader, test_loader = create_dataloaders(
            heat1d_path,
            batch_size=32,
            input_steps=10,
            output_steps=10
        )
        
        # Test iteration
        for batch in train_loader:
            print("\nBatch shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
            break