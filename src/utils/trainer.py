import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import os

def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=100,
    lr=1e-3,
    optimizer=None,
    lr_scheduler=None,
    device="cpu",
    save_path=None,
    verbose=True,
    loss_fn=None
):
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer['name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer}")
    
    if lr_scheduler is None:
        lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.97)
    elif lr_scheduler['name'] == 'StepLR':
        lr_scheduler = StepLR(optimizer, step_size=lr_scheduler['step_size'], gamma=lr_scheduler['gamma'])
    elif lr_scheduler['name'] == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"不支持的学习率调度器类型: {lr_scheduler}")

    # 初始化损失记录列表和最佳验证损失
    train_losses = []
    train_losses_recon = []
    train_losses_pred = []
    train_losses_latent = []
    train_losses_multi_step = []
    val_losses = []
    val_losses_recon = []
    val_losses_pred = []
    val_losses_latent = []
    val_losses_multi_step = []
    best_val_loss = float('inf')

    # 创建epoch级别的进度条
    epoch_pbar = tqdm(range(epochs), desc='Training', leave=True)
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_loss_recon = 0.0
        train_loss_pred = 0.0
        train_loss_latent = 0.0
        train_loss_multi_step = 0.0

        # 创建batch级别的进度条
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for x_t, x_tp1 in batch_pbar:
            x_t = x_t.to(device)
            x_tp1 = x_tp1.to(device)

            x_t = x_t.view(x_t.size(0), -1)
            x_tp1 = x_tp1.view(x_tp1.size(0), -1)

            output = model(x_t, x_tp1)
            loss_recon, loss_pred, loss_latent, loss_multi_step = loss_fn(output, x_t, x_tp1, multi_step=5)
            loss = loss_recon + loss_pred + loss_latent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_t.size(0)
            train_loss_recon += loss_recon.item() * x_t.size(0)
            train_loss_pred += loss_pred.item() * x_t.size(0)
            train_loss_latent += loss_latent.item() * x_t.size(0)
            # train_loss_multi_step += loss_multi_step.item() * x_t.size(0)

            # 更新batch进度条
            batch_pbar.set_postfix({'batch_loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader.dataset)
        train_loss_recon /= len(train_loader.dataset)
        train_loss_pred /= len(train_loader.dataset)
        train_loss_latent /= len(train_loader.dataset)
        # train_loss_multi_step /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_losses_recon.append(train_loss_recon)
        train_losses_pred.append(train_loss_pred)
        train_losses_latent.append(train_loss_latent)
        # train_losses_multi_step.append(train_loss_multi_step)

        # 在每个epoch结束时更新学习率
        lr_scheduler.step()
        
        # --- Validation ---
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_loss_recon = 0.0
            val_loss_pred = 0.0
            val_loss_latent = 0.0
            val_loss_multi_step = 0.0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc='Validation', leave=False)
                for x_t, x_tp1 in val_pbar:
                    x_t = x_t.to(device)
                    x_tp1 = x_tp1.to(device)

                    x_t = x_t.view(x_t.size(0), -1)
                    x_tp1 = x_tp1.view(x_tp1.size(0), -1)

                    output = model(x_t, x_tp1)
                    loss_recon, loss_pred, loss_latent, loss_multi_step = loss_fn(output, x_t, x_tp1, multi_step=5)
                    loss = loss_recon + loss_pred + loss_latent
                    val_loss += loss.item() * x_t.size(0)
                    val_loss_recon += loss_recon.item() * x_t.size(0)
                    val_loss_pred += loss_pred.item() * x_t.size(0)
                    val_loss_latent += loss_latent.item() * x_t.size(0)
                    # val_loss_multi_step += loss_multi_step.item() * x_t.size(0)

                    # 更新验证进度条
                    val_pbar.set_postfix({'batch_loss': f'{loss.item():.6f}'})

            val_loss /= len(val_loader.dataset)
            val_loss_recon /= len(val_loader.dataset)
            val_loss_pred /= len(val_loader.dataset)
            val_loss_latent /= len(val_loader.dataset)
            # val_loss_multi_step /= len(val_loader.dataset)
            val_losses.append(val_loss)
            val_losses_recon.append(val_loss_recon)
            val_losses_pred.append(val_loss_pred)
            val_losses_latent.append(val_loss_latent)
            # val_losses_multi_step.append(val_loss_multi_step)
            # 更新epoch进度条信息
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'val_loss_recon': f'{val_loss_recon:.6f}',
                'val_loss_pred': f'{val_loss_pred:.6f}',
                'val_loss_latent': f'{val_loss_latent:.6f}',
                # 'val_loss_multi_step': f'{val_loss_multi_step:.6f}'
            })

            # 保存最佳模型
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }, save_path)
                epoch_pbar.write(f'Epoch {epoch+1}: New best model saved! (val_loss: {val_loss:.6f})')

    if save_path:
        print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
        print(f"最佳模型保存路径: {save_path}")

    return train_losses, val_losses
