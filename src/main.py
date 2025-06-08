import argparse
import yaml
from models.koopman_ae import KoopmanAE
from utils.dataloader import lorenz63_dataloader, kolmogorov_dataloader_npy
from utils.trainer import train_model
from utils.lossfn import koopman_loss
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_path, model_name, dataset_name):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config[dataset_name]

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练Koopman模型')
    parser.add_argument('--model', type=str, default='KoopmanAE', help='模型名称')
    parser.add_argument('--dataset', type=str, default='lorenz63', help='数据集名称')
    parser.add_argument('--config', type=str, default='configs/KoopmanAE.yaml', help='配置文件路径')
    parser.add_argument('--show_figure', action='store_true', default=False, help='是否显示损失曲线图')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config, args.model, args.dataset)

    # 加载数据
    if args.dataset == 'lorenz63':
        train_loader, val_loader = lorenz63_dataloader(
            config['data_path'], 
            config['batch_size'], 
            config['train_ratio'], 
            config['val_ratio']
        )
        # 获取输入维度
        input_dim = next(iter(train_loader))[0].shape[1]
        print("Lorenz63 batch形状:",next(iter(train_loader))[0].shape)
        print(f"Lorenz63 输入维度: {input_dim}")
    elif args.dataset == 'kolmogorov':
        train_loader, val_loader = kolmogorov_dataloader_npy(
            config['data_path'], 
            config['batch_size'], 
            config['train_ratio'], 
            config['val_ratio']
        )
        # 获取输入维度
        input_dim = next(iter(train_loader))[0].shape[1] * next(iter(train_loader))[0].shape[2] * next(iter(train_loader))[0].shape[3]  # [16, 64, 64, 2]
        print("Kolmogorov batch形状:",next(iter(train_loader))[0].shape)
        print(f"Kolmogorov 输入维度: {input_dim}")

    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset}")

    # 初始化模型
    if args.model == 'KoopmanAE':
        model = KoopmanAE(
            input_dim=input_dim,
            latent_dim=config['latent_dim']
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    # 训练模型
    losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        optimizer=config['optimizer'],
        lr_scheduler=config['lr_scheduler'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path=config['save_path'],
        loss_fn=koopman_loss
    )

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses[0], label="Train Loss")
    plt.plot(losses[1], label="Val Loss")
    plt.title(f"{args.model} on {args.dataset}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{args.model}_{args.dataset}_loss.png")

if __name__ == "__main__":
    main()
