"""
Classes for interfacing predictive time series models with data and reinforcement learners.

A predictor consumes a series of observations (images) (batch_size, series_len, H, W, C), actions and rewards.
Then outputs:
    * a series of predicted observations (batch_size, series_len, H, W, C)
    * a series of predicted rewards (batch_size, series_len)

Requirements:
1. Encapsulate different stages of prediction
    * image encoding
    * action encoding
    * physics propagation
    * image decoding
    * reward decoding
    * state sampling
    * state inference (required for learning)

2. List of compatible neural models
    * predictive AE
    * predictive VAE
    * Conv RNN
    * VAE + MDN (World Models)
    * I2A and LaQ models
    * PlaNet
    * SLAC

3. Visualisations and plotting?
    * plotly training report
    * generate gifs

4. Functions
    * warm-up
    * predict next obs
    * predict n-next obs
    * evaluate obs likelihood

"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from .losses import normal_KL_div, gaussian_mix_nll
from .simple_models import StatePropagator, ActionStatePropagator, SigmoidFF

from .vae_wm import Encoder, Decoder, MixtureDensityNet


class Predictor(nn.Module):
    """
    Abstract class.
    """
    def __init__(self, **models):
        super(Predictor, self).__init__()
        self.encoding_is_deterministic = None
        self.decoding_is_deterministic = None
        self.transition_is_determininstic = None

        self.action_space = None

        self.initial_observations = None
        self.measurement_update_probability = None
        self.skip_null_action = None

        self.image_encoder = models.get("image_encoder", None)
        self.action_encoder = models.get("action_encoder", None)
        self.image_decoder = models.get("image_decoder", None)
        self.reward_decoder = models.get("reward_decoder", None)
        self.measurement_updater = models.get("measurement_updater", None)
        self.action_propagator = models.get("action_propagator", None)
        self.env_propagator = models.get("env_propagator", None)


class VAE_MDN(nn.Module):
    def __init__(self, image_channels, action_space, latent_size=64, n_gaussians=5, models={}):
        super(VAE_MDN, self).__init__()

        self.encoding_is_deterministic = False
        self.decoding_is_deterministic = True
        self.transition_is_determininstic = False

        self.action_space = action_space

        self.initial_observations = 5
        self.update_probability = 1.0
        self.skip_null_action = False

        self.latent_size = latent_size
        self.action_size = action_space
        self.image_channels = image_channels
        self.n_gaussians = n_gaussians

        self.image_encoder = models.get("image_encoder", Encoder(image_channels, latent_size, deterministic=self.encoding_is_deterministic))
        # TODO convert action to dim=1 continuous
        # self.action_encoder = models.get("action_encoder", None)
        self.image_decoder = models.get("image_decoder", Decoder(image_channels, latent_size))
        self.reward_decoder = models.get("reward_decoder",  nn.Linear(latent_size, 1))
        self.done_decoder = models.get("done_decoder",  SigmoidFF)
        # self.measurement_updater = models.get("measurement_updater", None)
        self.action_propagator = models.get("action_propagator", ActionStatePropagator(latent_size, latent_size, self.action_size))
        self.env_propagator = models.get("env_propagator", MixtureDensityNet(latent_size, latent_size, n_gaussians))

    def reconstruct_unordered(self, obs):
        z, mu, logsigma = self.image_encoder(obs)
        recon_x = self.image_decoder(z)
        return recon_x, mu, logsigma

    def get_vae_loss(self, recon_obs, obs_in, mu, logsigma):
        # TODO use averages or full?
        # reconstruction:
        # squared error loss per pixel per channel
        recon_loss = F.mse_loss(recon_obs, obs_in, size_average=False)

        # alternatively likelihood loss
        # TODO add likelihood function

        # variational loss per dimension of latent code
        KLD = normal_KL_div(mu, logsigma)
        total_loss = KLD + recon_loss
        return dict(reconstruction=recon_loss, variational=KLD, total=total_loss)

    def get_prediction_loss(self, o_series, o_next_series, a_series, reward_series, done_series, return_recons=False):
        batch_size = o_series.size(0)
        series_len = o_series.size(1)

        memory = None
        # memory = 0.5 * torch.ones(1, batch_size, 64).to(o_series.device)
        o_enc_preds = []
        # o_recons = []
        o_predictions = []
        r_predictions = []
        done_predictions = []
        latent_loss = []
        for t in range(series_len):
            o_0 = o_series[:, t, ...]
            o_1 = o_next_series[:, t, ...]
            a_0 = a_series[:, t, ...].squeeze()
            done_1 = done_series[:, t, ...]

            with torch.no_grad():
                o_0_enc, _, _ = self.image_encoder(o_0)
                o_1_enc, _, _ = self.image_encoder(o_1)

            belief, memory = self.action_propagator(a_0, o_0_enc, memory)
            z_mu, z_sigma, z_logpi = self.env_propagator(belief)

            memory[:, done_1.squeeze(), ...] = 0.0

            # don't compute loss for initial obs (warm up period)
            if t >= self.initial_observations:
                latent_loss.append(gaussian_mix_nll(o_1_enc, z_mu, z_sigma, z_logpi))

            if return_recons:
                with torch.no_grad():
                    z_next = self.env_propagator.get_sample(z_logpi, z_mu, z_sigma)
                    o_pred = self.image_decoder(z_next)
                    o_predictions.append(o_pred)

        latent_loss = torch.mean(torch.stack(latent_loss))
        total_loss = latent_loss
        losses = dict(latent_nll=latent_loss, total=total_loss)
        if return_recons:
            o_predictions = torch.stack(o_predictions, dim=1)
        return losses, o_predictions

    def free_running_prediction(self, o_series, a_series, deterministic=False):
        batch_size = o_series.size(0)
        series_len = o_series.size(1)

        memory = None
        o_enc_preds = []
        # o_recons = []
        o_predictions = []
        r_predictions = []
        done_predictions = []
        latent_loss = []
        with torch.no_grad():
            for t in range(series_len):
                o_0 = o_series[:, t, ...]
                a_0 = a_series[:, t, ...].squeeze()

                if t < self.initial_observations:
                    o_0_enc, _, _ = self.image_encoder(o_0)
                else:
                    o_0_enc = z_next

                belief, memory = self.action_propagator(a_0, o_0_enc, memory)
                z_mu, z_sigma, z_logpi = self.env_propagator(belief)

                z_next = self.env_propagator.get_sample(z_logpi, z_mu, z_sigma, deterministic)
                o_pred = self.image_decoder(z_next)
                o_predictions.append(o_pred)

        o_predictions = torch.stack(o_predictions, dim=1)
        return o_predictions


if __name__ == "__main__":
    pass