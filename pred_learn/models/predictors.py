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

        self.image_encoder = models.get("image_encoder", None)
        self.action_encoder = models.get("action_encoder", None)
        self.image_decoder = models.get("image_decoder", None)
        self.reward_decoder = models.get("reward_decoder", None)
        self.measurement_updater = models.get("measurement_updater", None)
        self.action_propagator = models.get("action_propagator", None)
        self.env_propagator = models.get("env_propagator", None)

    def forward(self, *tensors):
        raise NotImplemented


class AE_Predictor(Predictor):
    def __init__(self, action_dim, models={}):
        super(AE_Predictor, self).__init__()

        from pred_learn.models.ae import Encoder, ActionEncoder, Decoder, BeliefStatePropagator

        self.transition_is_determininstic = True
        self.observation_is_determininstic = True
        self.reward_is_determininstic = True

        self.skip_update_p = 0.5
        self.skip_null_action = True

        # mse or likelihood
        self.recon_loss = None
        # VAE or beta-VAE or other
        self.regularisation_loss = None

        self.image_encoder = models.get("image_encoder", Encoder())
        self.action_encoder = models.get("action_encoder", ActionEncoder(action_dim))
        self.image_decoder = models.get("image_decoder", Decoder())

        self.measurement_updater = models.get("measurement_updater", BeliefStatePropagator())
        self.action_propagator = models.get("action_propagator", BeliefStatePropagator())

        self.env_propagator = models.get("state_propagator", BeliefStatePropagator())

    def generate_predictions(self, o_series, a_series):
        batch_size = o_series.size(0)
        series_len = o_series.size(1)
        belief = None
        o_recons = []
        o_predictions = []
        for t in range(series_len):
            o_0 = o_series[:, t, ...]
            a_0 = a_series[:, t, ...]
            # r_t = r_series[:, t, ...]

            o_0_enc = self.image_encoder(o_0).unsqueeze(1)
            # TODO add masking of o_t_enc (skip for initial ts)
            if t < 3:
                out, belief = self.measurement_updater(o_0_enc, belief)
            else:
                out, updated_belief = self.measurement_updater(o_0_enc, belief)

                mask = torch.ones(updated_belief.size(), device=o_0.device)
                skip = torch.ByteTensor(np.random.rand(batch_size) < self.skip_update_p).cuda()
                mask[:, skip, ...] = 0
                belief = mask * updated_belief + (1 - mask) * belief

            o_recon = self.image_decoder(out)
            o_recons.append(o_recon.unsqueeze(1))

            a_0_enc = self.action_encoder(a_0).unsqueeze(1)
            out, updated_belief = self.action_propagator(a_0_enc, belief)
            # TODO add masking of null actions
            if self.skip_null_action:
                mask = torch.ones(updated_belief.size(), device=o_0.device)
                skip = (a_0 == 0).view(-1).cuda().byte()
                mask[:, skip, ...] = 0
                belief = mask * updated_belief + (1 - mask) * belief

            out, belief = self.env_propagator(out, belief)
            o_prediction = self.image_decoder(out)
            o_predictions.append(o_prediction.unsqueeze(1))

        o_recons = torch.cat(o_recons, dim=1)
        o_predictions = torch.cat(o_predictions, dim=1)
        return o_recons, o_predictions

    def forward(self, *tensors):
        assert 0 < len(tensors) <= 2

        if len(tensors) == 1:
            xs = tensors[0]
            h = None
        else:
            xs, h = tensors

        out_im = []
        for step in range(xs.size(1)):
            x = xs[:, step, ...]
            z = self.image_encoder(x)
            next_z, h = self.env_propagator(z.unsqueeze(0), h)
            next_x = self.decoder(next_z.squeeze(0))
            out_im.append(next_x.unsqueeze(1))

        x_preds = torch.cat(out_im, dim=1)
        return x_preds


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
