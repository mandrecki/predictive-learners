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


def gaussian_likelihood(target, mu, sigma):
    return 1/sigma**2 * torch.exp(-(target - mu)**2/sigma**2)

def gaussian_log_likelihood(target, mu, sigma):
    return -torch.log(sigma**2) - (target - mu)**2/sigma**2

def gaussian_nll_loss(target, mu, sigma):
    return torch.mean(-gaussian_log_likelihood(target, mu, sigma))

def guassian_mix_ll(target, mu, sigma, pi):
    pi_ps = torch.softmax(pi, dim=-1)
    gaussian_log_ps = gaussian_log_likelihood(target.unsqueeze(-1), mu, sigma)
    max_gaussian_log_ps = gaussian_log_ps.max(dim=-1)[0]
    mix_ps = pi_ps * torch.exp(gaussian_log_ps - max_gaussian_log_ps.unsqueeze(-1))
    mix_log_ps = torch.log(torch.sum(mix_ps, dim=-1)) + 2 * max_gaussian_log_ps
    mix_neg_log_loss = torch.mean(-mix_log_ps)
    return mix_neg_log_loss

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
        self.action_space = None

        self.update_probability = 0.0
        self.skip_null_action = False

        self.image_encoder = models.get("image_encoder", None)
        self.action_encoder = models.get("action_encoder", None)
        self.image_decoder = models.get("image_decoder", None)
        self.reward_decoder = models.get("reward_decoder", None)
        self.measurement_updater = models.get("measurement_updater", None)
        self.action_propagator = models.get("action_propagator", None)
        self.env_propagator = models.get("env_propagator", None)

    def reconstruction_loss(self, prediction, target):
        raise NotImplemented

    def regularisation_loss(self, x):
        raise NotImplemented

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
            skip = (torch.rand(updated_belief.size(1)) < self.update_probability).byte().cuda()
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

            update_probability = self.update_probability if t > 3 else 1
            belief_out, belief = self.update_belief_maybe(o_0, belief, update_probability)  # deterministic

            # o_0_recon = self.decode_belief(belief.view(batch_size, -1))
            # o_recons.append(o_0_recon.unsqueeze(1))

            belief_out, belief = self.propagate_action_conseq(belief, a_0, self.skip_null_action)  # possibly stochastic
            if self.skip_null_action:
                belief = self.propagate_env_conseq(belief)

            o_1_pred = self.decode_belief(belief.view(batch_size, -1))
            o_predictions.append(o_1_pred.unsqueeze(1))

        o_recons = torch.cat(o_recons, dim=1)
        o_predictions = torch.cat(o_predictions, dim=1)
        return o_recons, o_predictions, belief




class VAE_MDN(Predictor):
    def __init__(self, image_channels, action_size, latent_size=64, models={}):
        super(VAE_MDN, self).__init__()
        # self.update_probability = 0.0
        # self.skip_null_action = False
        self.latent_size = latent_size
        self.action_size = action_size
        self.image_channels = image_channels

        from .vae_wm import Encoder, Decoder, MDRNN

        self.image_encoder = models.get("image_encoder", Encoder(image_channels, latent_size))
        self.action_encoder = models.get("action_encoder", None)
        self.image_decoder = models.get("image_decoder", Decoder(image_channels, latent_size))
        self.reward_decoder = models.get("reward_decoder", None)
        self.measurement_updater = models.get("measurement_updater", None)
        self.action_propagator = models.get("action_propagator", MDRNN(latent_size, action_size, latent_size, 5))
        # self.env_propagator = models.get("env_propagator", None)

    def reconstruct_unordered(self, obs):
        mu, logsigma = self.image_encoder(obs)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.image_decoder(z)
        return recon_x, mu, logsigma

    def to_latent(self, obs, next_obs):
        """ Transform observations to latent space.
        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        batch_size = obs.size()[0]
        series_len = obs.size()[1]
        with torch.no_grad():
            obs, next_obs = [
                x.view(-1, self.image_channels, 64, 64) for x in (obs, next_obs)]

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                self.image_encoder(x) for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(batch_size, series_len, self.latent_size)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        return latent_obs, latent_next_obs

    def get_loss(self, latent_obs, action, reward, terminal,
                 latent_next_obs, include_reward: bool):
        """ Compute losses.
        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).
        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        batch_size = latent_obs.size()[0]
        series_len = latent_obs.size()[1]
        latent_obs, action, \
        reward, terminal, \
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
        mus, sigmas, logpi, rs, ds = self.action_propagator(action, latent_obs)
        gmm = self.gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = F.binary_cross_entropy_with_logits(ds, terminal.squeeze(-1))
        if include_reward:
            mse = F.mse_loss(rs, reward)
            scale = series_len + 2
        else:
            mse = 0
            scale = series_len + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

    def predict_series(self, o_series, a_series):
        batch_size = o_series.size(0)
        series_len = o_series.size(1)

        # get latents (vectorised)
        # o_series = o_series.view(-1, o_series.size()[2:])
        # latents = self.image_encoder(o_series).view(batch_size, series_len, -1)
        # mus, sigmas, logpi, rs, ds = self.action_propagator(latents, a_series)

        belief = None
        o_enc_preds = []
        # o_recons = []
        o_predictions = []
        r_predictions = []
        done_predictions = []
        other = {
            'vae_mus': [],
            'vae_logsigmas': []

        }
        for t in range(series_len):
            o_0 = o_series[:, t, ...]
            a_0 = a_series[:, t, ...]

            mu, logsigma = self.image_encoder(o_0)
            o_0_enc = torch.randn_like(logsigma.exp()).add(mu)
            mu, logsigma = self.image_encoder(o_1)
            o_0_enc = torch.randn_like(logsigma.exp()).add(mu)


            mus, sigmas, logpi, rs, ds = self.propagate_all(belief, a_0, o_0_enc)  # possibly stochastic

            o_1_pred = self.decode_belief(belief.view(batch_size, -1))
            o_predictions.append(o_1_pred.unsqueeze(1))

        o_recons = torch.cat(o_recons, dim=1)
        o_predictions = torch.cat(o_predictions, dim=1)
        # return o_recons, o_predictions, belief

        return o_enc_preds,

    def loss(self, recon, target, mu, logsigma):
        MSE = F.mse_loss(recon, target, size_average=False)

        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return MSE + KLD

    def get_vae_loss(self, obs_in, compute_gradients=True):
        if compute_gradients:
            obs_recon, mu, logsigma = self.reconstruct_unordered(obs_in)
            loss = self.loss(obs_recon, obs_in, mu, logsigma)
        else:
            with torch.no_grad():
                obs_recon, mu, logsigma = self.reconstruct_unordered(obs_in)
                loss = self.loss(obs_recon, obs_in, mu, logsigma)
        return loss

    def gmm_loss(self, batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
        """ Computes the gmm loss.
        Compute minus the log probability of batch under the GMM model described
        by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
        dimensions (several batch dimension are useful when you have both a batch
        axis and a time step axis), gs the number of mixtures and fs the number of
        features.
        :args batch: (bs1, bs2, *, fs) torch tensor
        :args mus: (bs1, bs2, *, gs, fs) torch tensor
        :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
        :args logpi: (bs1, bs2, *, gs) torch tensor
        :args reduce: if not reduce, the mean in the following formula is ommited
        :returns:
        loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
            sum_{k=1..gs} pi[i1, i2, ..., k] * N(
                batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
        NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
        with fs).
        """
        batch = batch.unsqueeze(-2)
        normal_dist = torch.distributions.normal.Normal(mus, sigmas)
        g_log_probs = normal_dist.log_prob(batch)
        g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
        max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
        g_log_probs = g_log_probs - max_log_probs

        g_probs = torch.exp(g_log_probs)
        probs = torch.sum(g_probs, dim=-1)

        log_prob = max_log_probs.squeeze() + torch.log(probs)
        if reduce:
            return - torch.mean(log_prob)
        return - log_prob

class AE_Predictor(Predictor):
    def __init__(self, image_channels, action_dim, models={}):
        super(AE_Predictor, self).__init__()

        from pred_learn.models.simple_models import Encoder, ActionEncoder, Decoder, StatePropagator, SigmoidFF

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

        self.measurement_updater = models.get("measurement_updater", StatePropagator())
        self.action_propagator = models.get("action_propagator", StatePropagator())

        self.env_propagator = models.get("state_propagator", SigmoidFF())

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
