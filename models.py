import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):

  def __init__(self, in_channels=1, latent_dim=10):
    super().__init__()
    self.in_channels = in_channels
    self.latent_dim = latent_dim

    self.encoder = nn.Sequential(
        # input: N, 1, 28, 28
        nn.Conv2d(1, 32, 3, stride=2, padding=1), # out: N, 16, 14, 14
        nn.SiLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), # out: N, 32, 7, 7
        nn.SiLU(),
        nn.Conv2d(64, latent_dim, 7), # out: N, latent_dim, 1, 1

        # additional layer in AE encoder to best approximate the additional linear layer present in VAE for mu and log_var
        nn.Flatten(start_dim=1), # must flatten the (1, 1) feature map dimension down to N, latent_dim
        nn.Linear(latent_dim, latent_dim)
     )

    self.decoder = nn.Sequential(
        # must unflatten the N, latent_dim input up to image space - N, latent_dim, 1, 1
        nn.Unflatten(-1, (latent_dim, 1, 1)),
        nn.ConvTranspose2d(latent_dim, 64, 7), # N, 32, 7, 7 (32 activations of 7x7)
        nn.SiLU(),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14
        nn.SiLU(),
        nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28
        nn.Sigmoid()
    )

  def encode(self, x):
    encoded = self.encoder(x)
    return encoded

  def get_loss(self, x, x_recon):
    # reconstruction loss
    recon_loss = F.mse_loss(x_recon, x)
    return recon_loss

  def forward(self, x):
    z = self.encode(x)
    x_recon = self.decoder(z)
    return z, x_recon
class VAE(nn.Module):

  def __init__(self, in_channels=1, latent_dim=10, kld_weight=0.00025):
    super().__init__()
    self.in_channels = in_channels
    self.latent_dim = latent_dim
    self.kld_weight = kld_weight

    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), # out: N, 16, 14, 14
        nn.SiLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), # out: N, 32, 7, 7
        nn.SiLU(),
        nn.Conv2d(64, latent_dim, 7), # out: N, latent_dim, 1, 1
        nn.Flatten(start_dim=1)
    )

    self.get_mu = nn.Linear(latent_dim, latent_dim)
    self.get_log_var = nn.Linear(latent_dim, latent_dim)

    self.decoder = nn.Sequential(
        nn.Unflatten(-1, (latent_dim, 1, 1)),
        nn.ConvTranspose2d(latent_dim, 64, 7), # N, 32, 7, 7 (32 activations of 7x7)
        nn.SiLU(),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14
        nn.SiLU(),
        nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28
        nn.Sigmoid()
    )

  def encode(self, x):
    encoded = self.encoder(x)
    mu = self.get_mu(encoded)
    log_var = self.get_log_var(encoded)
    return mu, log_var

  def get_loss(self, x, x_recon, mu, log_var):
    # KL divergence loss
    kld = torch.mean(-0.5 * torch.sum(log_var - mu ** 2 - log_var.exp() + 1, dim=1), dim=0)

    # reconstruction loss
    recon_loss = F.mse_loss(x_recon, x)

    # total loss
    loss = recon_loss + self.kld_weight * kld
    return loss

  def reparameterize(self, mu, log_var):
    epsilon = torch.randn_like(log_var)
    std_dev = 0.5 * log_var.exp()
    z = mu + epsilon * std_dev
    return z

  def forward(self, x):
    mu, log_var = self.encode(x)
    z = self.reparameterize(mu, log_var)
    x_recon = self.decoder(z)
    return z, x_recon, mu, log_var
