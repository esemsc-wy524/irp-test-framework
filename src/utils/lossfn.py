import torch
import torch.nn.functional as F

def koopman_loss(output, x_t, x_tp1, multi_step=1):
    multi_step_loss = 0.0

    # 1. Reconstruction loss
    loss_recon = F.mse_loss(output["x_t_recon"], x_t)

    # 2. One-step prediction loss in original space
    loss_pred = F.mse_loss(output["x_tp1_pred"], x_tp1)

    # 3. Koopman latent consistency loss
    loss_latent = F.mse_loss(output["z_tp1_pred"], output["z_tp1_true"])

    # 4. Multi-step prediction stability loss
    z = output["z_t"].detach()  # latent start point
    K = output["K"]
    decoder = output["decoder"]
    x_pred_prev = output["x_tp1_pred"].detach()  # first step prediction

    for k in range(2, multi_step + 1):
        # Latent dynamics propagation
        z = torch.matmul(z, K.T)
        x_pred_k = decoder(z)

        # loss: predict x_k vs previous prediction x_{k-1}
        loss_pred_k = F.mse_loss(x_pred_k, x_pred_prev)
        multi_step_loss += loss_pred_k

        # update for next prediction target
        x_pred_prev = x_pred_k.detach()

    return loss_recon, loss_pred, loss_latent, multi_step_loss
