import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=10, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        # Learnable Koopman matrix
        self.K = nn.Parameter(torch.randn(latent_dim, latent_dim))

    def forward(self, z):
        return z @ self.K.T

class KoopmanAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.koopman = KoopmanOperator(latent_dim)

    def forward(self, x_t, x_tp1):
        # Encode current and next state

        z_t = self.encoder(x_t)         # latent at time t
        z_tp1_true = self.encoder(x_tp1)  # latent at time t+1
        z_tp1_pred = self.koopman(z_t)    # predicted latent at t+1

        # Decode predicted latent
        x_tp1_pred = self.decoder(z_tp1_pred)
        x_t_recon = self.decoder(z_t)

        return {
            "z_t": z_t,
            "z_tp1_true": z_tp1_true,
            "z_tp1_pred": z_tp1_pred,
            "x_t_recon": x_t_recon,
            "x_tp1_pred": x_tp1_pred,
            "K": self.koopman.K,
            "decoder": self.decoder
        }
    
    def predict(self, x_t):
        z_t = self.encoder(x_t)
        z_tp1_pred = self.koopman(z_t)
        x_tp1_pred = self.decoder(z_tp1_pred)
        return x_tp1_pred
