import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.koopman_ae import KoopmanAE

def autoregressive_predict(model, x0, steps, device):
    """
    使用模型进行自回归预测
    - x0: 初始输入帧，shape = (1, 1, 64, 64)
    - steps: 向后预测的帧数
    - 返回 shape = (steps, 64, 64)
    """
    preds = []
    xt = x0.to(device)
    xt = xt.reshape(xt.shape[0], -1)
    for _ in range(steps):
        xt = model.predict(xt)
        preds.append(xt.detach().numpy())  # 去掉 batch & channel
    return np.array(preds)

def visualize_prediction_vs_truth(model_path, data_path, sample_index=10, steps=50, output_gif="comparison.gif"):
    # === 加载数据 ===
    data = np.load(data_path)  # shape: (120, 320, 64, 64)
    sample = data[sample_index]  # shape: (320, 64, 64)
    gt_seq = sample[:steps]  # 取前 steps 帧作为 ground truth

    # === 初始帧（1, 1, H, W） ===
    x0 = torch.tensor(gt_seq[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)

    # === 加载模型 ===
    input_dim = 64 * 64
    latent_dim = 32  # 替换为你的实际值
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = KoopmanAE(input_dim=input_dim, latent_dim=latent_dim)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # === 执行自回归预测 ===
    pred_seq = autoregressive_predict(model, x0, steps, device)
    # print(pred_seq.shape)

    pred_seq = pred_seq.reshape(steps, 64, 64)
    # print(pred_seq.shape)

    # === 创建动图 ===
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ims = []

    im_pred = axs[0].imshow(pred_seq[0], cmap='viridis', origin='lower')
    axs[0].set_title("Model Prediction")

    im_gt = axs[1].imshow(gt_seq[0], cmap='viridis', origin='lower')
    axs[1].set_title("Ground Truth")

    for t in range(steps):
        im_pred = axs[0].imshow(pred_seq[t], cmap='viridis', origin='lower', animated=True)
        im_gt = axs[1].imshow(gt_seq[t], cmap='viridis', origin='lower', animated=True)
        ims.append([im_pred, im_gt])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    ani.save(output_gif, writer='pillow')
    print(f"[INFO] 动图已保存为 {output_gif}")


if __name__ == "__main__":
    visualize_prediction_vs_truth(
        model_path='results/checkpoints/KoopmanAE_kolmogorov.pt',
        data_path='data/kolmogorov/RE1000/kf_2d_re1000_64_120seed.npy',
        sample_index=10,
        steps=50
    )