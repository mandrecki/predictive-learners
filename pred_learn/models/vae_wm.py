
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size, deterministic=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.deterministic = deterministic
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        if self.deterministic is False:
            self.fc_logsigma = nn.Linear(2*2*256, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        if self.deterministic is False:
            logsigma = self.fc_logsigma(x)
            sigma = logsigma.exp()
            eps = torch.randn_like(sigma)
            z = eps.mul(sigma).add_(mu)
        else:
            z = mu
            logsigma = None

        return z, mu, logsigma


class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        z, mu, logsigma = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


class MixtureDensityNet(nn.Module):
    """ MDRNN model for multi steps forward """
    def __init__(self, in_size, latents, n_gaussians):
        super().__init__()
        self.in_size = in_size
        self.latents = latents
        self.n_gaussians = n_gaussians

        self.fc_logpi = nn.Linear(in_size, n_gaussians)
        self.fc_logpi = nn.Sequential(
            nn.Linear(in_size, n_gaussians),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(in_size, latents * n_gaussians),
        )
        self.fc_logsigma = nn.Sequential(
            nn.Linear(in_size, latents * n_gaussians),
        )

    def forward(self, x, return_sample=False):
        batch_size = x.size(0)

        mu = self.fc_mu(x)
        mu = mu.view(batch_size, self.n_gaussians, self.latents)

        logsigma = self.fc_logsigma(x)
        logsigma = logsigma.view(batch_size, self.n_gaussians, self.latents)
        sigma = torch.exp(logsigma)

        logpi = self.fc_logpi(x)
        logpi = logpi.view(batch_size, self.n_gaussians)
        logpi = F.log_softmax(logpi, dim=-1)

        return mu, sigma, logpi

    def get_sample(self, logpi, mu, sigma):
        with torch.no_grad():
            batch_size = logpi.size(0)
            pi_dist = Categorical(probs=logpi.exp())
            draw = pi_dist.sample()
            mu_drawn = mu[torch.arange(batch_size), draw, ...]
            sigma_drawn = sigma[torch.arange(batch_size), draw, ...]
            eps = torch.randn_like(sigma_drawn)
            z = eps.mul(sigma_drawn).add_(mu_drawn)
            return z

