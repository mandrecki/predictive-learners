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
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F


import numpy as np
import gym

from .losses import normal_KL_div, gaussian_mix_nll
from .simple_models import StatePropagator, ActionStatePropagator, SigmoidFF , LinearFF

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
        if type(action_space) is gym.spaces.discrete.Discrete:
            self.action_size = action_space.n
        elif type(action_space) is gym.spaces.Box:
            self.action_size = action_space.shape[0]
        else:
            raise ValueError("Bad action space type given to model: {}".format(type(action_space)))

        self.initial_observations = 5
        self.update_probability = 1.0
        self.skip_null_action = False

        self.latent_size = latent_size
        self.image_channels = image_channels
        self.n_gaussians = n_gaussians

        self.image_encoder = models.get("image_encoder", Encoder(image_channels, latent_size, deterministic=self.encoding_is_deterministic))
        # TODO convert action to dim=1 continuous
        # self.action_encoder = models.get("action_encoder", None)
        self.image_decoder = models.get("image_decoder", Decoder(image_channels, latent_size))
        self.reward_decoder = models.get("reward_decoder",  LinearFF(latent_size, 1))
        self.done_decoder = models.get("done_decoder",  SigmoidFF(latent_size, 1))
        # self.measurement_updater = models.get("measurement_updater", None)
        self.action_propagator = models.get("action_propagator", ActionStatePropagator(latent_size, latent_size, self.action_size))
        self.env_propagator = models.get("env_propagator", MixtureDensityNet(latent_size, latent_size, n_gaussians))

    def reconstruct_unordered(self, obs):
        z, mu, logsigma = self.image_encoder(obs)
        recon_x = self.image_decoder(z)
        return recon_x, mu, logsigma

    def get_vae_loss(self, recon_obs, obs_in, mu, logsigma, free_nats=None):
        # TODO use averages or full?
        # reconstruction:
        # squared error loss per pixel per channel
        recon_loss = F.mse_loss(recon_obs, obs_in, size_average=False)

        # alternatively likelihood loss
        # TODO add likelihood function

        # variational loss per dimension of latent code
        KLD = normal_KL_div(mu, logsigma)
        if free_nats is not None:
            free_nats = torch.full((1,), free_nats).to(recon_obs.device)
            KLD = torch.max(KLD, free_nats)

        total_loss = KLD + recon_loss
        return dict(reconstruction=recon_loss, variational=KLD, total=total_loss)

    def get_prediction_loss(self, o_series, o_next_series, a_series, reward_series, done_series, return_recons=False):
        batch_size = o_series.size(0)
        series_len = o_series.size(1)

        memory = None
        o_predictions = []
        losses = {key: [] for key in ["latent", "reward", "done"]}
        for t in range(series_len):
            o0 = o_series[:, t, ...]
            o1 = o_next_series[:, t, ...]
            a0 = a_series[:, t, ...]
            r1 = reward_series[:, t, ...]
            done1 = done_series[:, t, ...]

            with torch.no_grad():
                o0_enc, _, _ = self.image_encoder(o0)
                o1_enc, _, _ = self.image_encoder(o1)

            belief, memory = self.action_propagator(a0, o0_enc, memory)
            z_mu, z_sigma, z_logpi = self.env_propagator(belief)
            rew_pred = self.reward_decoder(belief)
            done_pred = self.done_decoder(belief)

            memory[:, done1.squeeze(), ...] = 0.0

            # don't compute loss for initial obs (warm up period)
            if t >= self.initial_observations:
                losses["latent"].append(gaussian_mix_nll(o1_enc, z_mu, z_sigma, z_logpi))
                losses["reward"].append(F.mse_loss(rew_pred, r1))
                losses["done"].append(F.binary_cross_entropy(done_pred, done1.float()))

            if return_recons:
                with torch.no_grad():
                    z_next = self.env_propagator.get_sample(z_logpi, z_mu, z_sigma)
                    o_pred = self.image_decoder(z_next)
                    o_predictions.append(o_pred)

        for key, value in losses.items():
            losses[key] = torch.mean(torch.stack(value))
        losses["total"] = sum(list(losses.values()))
        if return_recons:
            o_predictions = torch.stack(o_predictions, dim=1)
        return losses, o_predictions

    def free_running_prediction(self, o_series, a_series, deterministic=False):
        batch_size = o_series.size(0)
        series_len = o_series.size(1)

        memory = None
        o_predictions = []
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


class PlaNetPredictor(Predictor):
    def __init__(self,  image_channels, action_space,
                 belief_size=200, state_size=30, hidden_size=200, embedding_size=1024,
                 activation_function='relu', min_std_dev=0.1):
        super(Predictor, self).__init__()
        models = {}

        self.image_channels = image_channels
        self.action_space = action_space
        self.belief_size = belief_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        if type(action_space) is gym.spaces.discrete.Discrete:
            self.action_size = action_space.n
        elif type(action_space) is gym.spaces.Box:
            self.action_size = action_space.shape[0]
        else:
            raise ValueError("Bad action space type given to model: {}".format(type(action_space)))

        from .planet_model import VisualObservationModel, VisualEncoder, TransitionModel, RewardModel, bottle
        self.bottle = bottle

        self.image_encoder = models.get("image_encoder", VisualEncoder(image_channels, embedding_size))
        self.image_decoder = models.get("image_decoder", VisualObservationModel(image_channels, belief_size, state_size, embedding_size))
        self.reward_decoder = models.get("reward_decoder", RewardModel(belief_size, state_size, hidden_size))
        self.transition_model = models.get("transition_model", TransitionModel(belief_size, state_size, self.action_size, hidden_size, embedding_size))

    def get_prediction_loss(self, o_series, o_next_series, a_series, reward_series, done_series, return_recons=False, free_nats=0):
        batch_size = o_series.size(1)
        series_len = o_series.size(0)
        device = o_series.device
        free_nats = torch.full((1,), free_nats, device=device)


        # losses = {key: [] for key in ["recon", "variational_encoding", "variational_transition", "reward", "done"]}
        # preds = {key: [] for key in ["recon", "belief", "reward", "done"]}

        nonterminals = (1 - done_series).float()
        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(batch_size, self.belief_size, device=device), torch.zeros(batch_size, self.state_size, device=device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, a_series[:-1], init_belief, self.bottle(self.image_encoder, (o_series[1:], )), nonterminals)
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        o_recons = self.bottle(self.image_decoder, (beliefs, posterior_states))
        recon_loss = F.mse_loss(o_recons, o_series[:-1], reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
        reward_pred = self.bottle(self.reward_decoder, (beliefs, posterior_states))
        reward_loss = F.mse_loss(reward_pred, reward_series[:-1].squeeze(), reduction='none').mean(dim=(0, 1))
        kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out

        total_loss = recon_loss + reward_loss + kl_loss
        losses = dict(total=total_loss, recon=recon_loss, variational=kl_loss, reward=reward_loss)
        preds = dict(recon=o_recons)
        return losses, preds

    def free_running_prediction(self, o_series, a_series, deterministic=False):
        batch_size = o_series.size(1)
        series_len = o_series.size(0)

        memory = None
        o_predictions = []
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



# class PlaNetPredictor(Predictor):
#     def __init__(self, image_channels, action_space, latent_size, **models):
#         super(PlaNetPredictor, self).__init__()
#         self.encoding_is_deterministic = True
#         self.decoding_is_deterministic = True
#         self.transition_is_determininstic = False
#         self.state_is_spatial = False
#
#         self.action_space = action_space
#         self.latent_size = latent_size
#
#         self.initial_observations = 3
#         self.measurement_update_probability = 1
#         self.skip_null_action = False
#
#         self.image_encoder = models.get("image_encoder", Encoder(image_channels, latent_size, deterministic=self.encoding_is_deterministic))
#         # TODO convert action to dim=1 continuous
#         # self.action_encoder = models.get("action_encoder", None)
#         self.image_decoder = models.get("image_decoder", Decoder(image_channels, latent_size))
#         self.reward_decoder = models.get("reward_decoder",  LinearFF(latent_size, 1))
#         self.done_decoder = models.get("done_decoder",  SigmoidFF(latent_size, 1))
#         self.measurement_updater = models.get("measurement_updater", nn.GRUCell(latent_size, 2*latent_size))
#         self.action_propagator = models.get("action_propagator", ActionStatePropagator(latent_size, latent_size, self.action_size))
#         # self.env_propagator = models.get("env_propagator", MixtureDensityNet(latent_size, latent_size, n_gaussians))
#
#     def get_series_prediction(self, o_series, o_next_series, a_series, reward_series, done_series,
#                               return_recons=False, overshooting=False):
#         preds = {key: [] for key in ["recon", "belief", "reward", "done"]}
#         losses = {key: [] for key in ["recon", "variational_encoding","variational_transition", "reward", "done"]}
#         batch_size = o_series.size(0)
#         series_len = o_series.size(1)
#         memory = None
#         for t in range(series_len):
#             o_0 = o_series[:, t, ...]
#             o_1 = o_next_series[:, t, ...]
#             a_0 = a_series[:, t, ...]
#
#             _, z_mu, _ = self.image_encoder(o_0)





if __name__ == "__main__":
    pass