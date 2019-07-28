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


class Predictor(nn.Module):
    """
    Abstract class.
    """
    def __init__(self, **models):
        super(Predictor, self).__init__()
        self.stateful = False

        self.transition_is_determininstic = None
        self.observation_is_determininstic = None
        self.reward_is_determininstic = None

        # mse or likelihood
        self.recon_loss = None
        # VAE or beta-VAE or other
        self.regularisation_loss = None

        self.action_space = None

        self.image_encoder = models.get("image_encoder", None)
        self.action_encoder = models.get("action_encoder", None)
        self.image_decoder = models.get("image_decoder", None)
        self.reward_decoder = models.get("reward_decoder", None)
        self.measurement_updater = models.get("measurement_updater", None)
        self.action_propagator = models.get("action_propagator", None)
        self.env_propagator = models.get("env_propagator", None)

    def encode_obs(self, obs):
        obs_enc = self.image_encoder(obs).unsqueeze(1)
        return obs_enc

    def update_belief_maybe(self, obs, belief, p=1):
        assert 0 <= p <= 1
        if belief is None:
            assert p == 1

        obs_enc = self.encode_obs(obs)
        if p == 1:
            out, belief = self.measurement_updater(obs_enc, belief)
        else:
            out, updated_belief = self.measurement_updater(obs_enc, belief)

            mask = torch.ones(updated_belief.size(), device=obs_enc.device)
            skip = (torch.rand(updated_belief.size(1)) < self.skip_update_p).byte().cuda()
            mask[:, skip, ...] = 0
            belief = mask * updated_belief + (1 - mask) * belief

        return out, belief

    def decode_belief(self, belief):
        obs_recon = self.image_decoder(belief)
        return obs_recon

    def encode_action(self, action):
        action_enc = self.action_encoder(action).unsqueeze(1)
        return action_enc

    def propagate_action_conseq(self, belief, action, skip_null_action=False):
        action_enc = self.encode_action(action)
        out, updated_belief = self.action_propagator(action_enc, belief)

        if skip_null_action:
            # TODO different null action test depending on action space
            skip_those = (action.view(-1) == 0).byte()
            mask = torch.ones(updated_belief.size(), device=updated_belief.device)
            mask[:, skip_those, ...] = 0
            belief = mask * updated_belief + (1 - mask) * belief
        else:
            belief = updated_belief
        return out, belief

    def propagate_env_conseq(self, belief):
        belief = self.env_propagator(belief.squeeze(0))
        return belief.unsqueeze(0)

    def predict_full(self, o_series, a_series):
        batch_size = o_series.size(0)
        series_len = o_series.size(1)
        belief = None
        o_recons = []
        o_predictions = []
        for t in range(series_len):
            o_0 = o_series[:, t, ...]
            a_0 = a_series[:, t, ...]

            update_probability = 0.9 if t > 3 else 1
            belief_out, belief = self.update_belief_maybe(o_0, belief, update_probability)  # deterministic

            o_0_recon = self.decode_belief(belief.view(batch_size, -1))
            o_recons.append(o_0_recon.unsqueeze(1))

            belief_out, belief = self.propagate_action_conseq(belief, a_0)  # possibly stochastic
            belief = self.propagate_env_conseq(belief)

            o_1_pred = self.decode_belief(belief.view(batch_size, -1))
            o_predictions.append(o_1_pred.unsqueeze(1))

        o_recons = torch.cat(o_recons, dim=1)
        o_predictions = torch.cat(o_predictions, dim=1)
        return o_recons, o_predictions, belief


    def forward(self, *tensors):
        raise NotImplemented


class AE_Predictor(Predictor):
    def __init__(self, image_channels, action_dim, models={}):
        super(AE_Predictor, self).__init__()

        from pred_learn.models.ae import Encoder, ActionEncoder, Decoder, BeliefStatePropagator, SimpleFF

        self.transition_is_determininstic = True
        self.observation_is_determininstic = True
        self.reward_is_determininstic = True

        self.skip_update_p = 0.5
        self.skip_null_action = True

        # mse or likelihood
        self.recon_loss = None
        # VAE or beta-VAE or other
        self.regularisation_loss = None

        self.image_encoder = models.get("image_encoder", Encoder(im_channels=image_channels))
        self.action_encoder = models.get("action_encoder", ActionEncoder(action_dim))
        self.image_decoder = models.get("image_decoder", Decoder(im_channels=image_channels))

        self.measurement_updater = models.get("measurement_updater", BeliefStatePropagator())
        self.action_propagator = models.get("action_propagator", BeliefStatePropagator())

        self.env_propagator = models.get("state_propagator", SimpleFF())

    def forward(self, x):
        return None


if __name__ == "__main__":
    import torch

    batch_size = 4
    series_len = 10
    h, w, c = 64, 64, 3

    image = torch.rand(batch_size, series_len, c, h, w)
    print(image.size())

    for predictor in [AE_Predictor]:
        p = predictor()
        print("prediction for ", predictor)
        print(p(image.size()))
