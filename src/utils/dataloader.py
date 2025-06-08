import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def lorenz63_dataloader(data_path, batch_size, train_ratio, val_ratio, shuffle=True):
    # 加载数据
    data = np.load(data_path).astype(np.float32)
    total_length = len(data) - 1  # 减1是因为我们需要预测下一个时间步
    
    # 计算训练集和验证集的大小
    train_size = int(total_length * train_ratio)
    val_size = int(total_length * val_ratio)
    
    # 划分数据集，确保输入和输出对齐
    # 训练集
    x_t_train = data[:train_size]  # 输入
    x_tp1_train = data[1:train_size + 1]  # 目标（下一时间步）
    
    # 验证集
    val_start = train_size
    val_end = train_size + val_size
    x_t_val = data[val_start:val_end]  # 输入
    x_tp1_val = data[val_start + 1:val_end + 1]  # 目标（下一时间步）
    
    # 转换为PyTorch张量
    x_t_train = torch.from_numpy(x_t_train)
    x_tp1_train = torch.from_numpy(x_tp1_train)
    x_t_val = torch.from_numpy(x_t_val)
    x_tp1_val = torch.from_numpy(x_tp1_val)
    
    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(x_t_train, x_tp1_train),
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    val_loader = DataLoader(
        TensorDataset(x_t_val, x_tp1_val),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def kolmogorov_dataloader_npy(npy_path, batch_size, train_ratio=0.8, val_ratio=0.2, shuffle=True):
    # 加载数据 (N_sample, T, H, W)
    data = np.load(npy_path)  # shape: (120, 320, 64, 64)
    data = data.astype(np.float32)

    N_sample, T, H, W = data.shape
    N_train = int(N_sample * train_ratio)
    N_val = int(N_sample * val_ratio)

    # 拆分样本集
    train_data = data[:N_train]         # shape: (N_train, T, H, W)
    val_data = data[N_train:N_train+N_val]  # shape: (N_val, T, H, W)

    # 构造 t -> t+1 对
    def make_pairs(arr):
        x_t = arr[:, :-1]    # (N, T-1, H, W)
        x_tp1 = arr[:, 1:]   # (N, T-1, H, W)
        x_t = x_t.reshape(-1, H, W)      # 合并样本和时间维: (N*(T-1), H, W)
        x_tp1 = x_tp1.reshape(-1, H, W)
        return x_t, x_tp1

    x_t_train, x_tp1_train = make_pairs(train_data)
    x_t_val, x_tp1_val = make_pairs(val_data)

    # 转成 tensor
    x_t_train = torch.from_numpy(x_t_train).unsqueeze(1)      # shape: (N*(T-1), 1, H, W)
    x_tp1_train = torch.from_numpy(x_tp1_train).unsqueeze(1)
    x_t_val = torch.from_numpy(x_t_val).unsqueeze(1)
    x_tp1_val = torch.from_numpy(x_tp1_val).unsqueeze(1)

    # 封装成 Dataloader
    train_loader = DataLoader(
        TensorDataset(x_t_train, x_tp1_train),
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        TensorDataset(x_t_val, x_tp1_val),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = kolmogorov_dataloader_npy(
        "./data/kolmogorov/RE1000/kf_2d_re1000_64_120seed.npy",
        batch_size=64,
        train_ratio=0.8,
        val_ratio=0.2,
        shuffle=True
    )
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)

    print(len(train_loader))
    print(len(val_loader))