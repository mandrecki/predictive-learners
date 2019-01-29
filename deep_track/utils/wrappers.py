import numpy as np
import gym
from gym.spaces import Box


class AddShuffleNoise1D(gym.ObservationWrapper):
    def __init__(self, env=None, n_repeats=1):
        super(AddShuffleNoise1D, self).__init__(env)
        self.n_repeats = n_repeats
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [n_repeats+1 * self.observation_space.shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        new_obs = [observation]
        for i in range(self.n_repeats):
            obs_shuffled = np.copy(observation)
            np.random.shuffle(obs_shuffled)
            new_obs.append(obs_shuffled)
        new_obs_np = np.concatenate(new_obs)
        return new_obs_np


class AddBigNoise1D(gym.ObservationWrapper):
    def __init__(self, env=None, n_repeats=1, bigness=100):
        super(AddBigNoise1D, self).__init__(env)
        self.n_repeats = n_repeats
        self.bigness = bigness
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [n_repeats+1 * self.observation_space.shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        new_obs = [observation]
        for i in range(self.n_repeats):
            noise = self.bigness * np.random.randn(*observation.shape)
            new_obs.append(noise)
        new_obs_np = np.concatenate(new_obs)
        return new_obs_np


class AddPadding1D(gym.ObservationWrapper):
    def __init__(self, env=None, n_repeats=1):
        super(AddPadding1D, self).__init__(env)
        self.n_repeats = n_repeats
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [n_repeats+1 * self.observation_space.shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        new_obs = [observation]
        for i in range(self.n_repeats):
            padding = np.zeros_like(observation)
            new_obs.append(padding)
        new_obs_np = np.concatenate(new_obs)
        return new_obs_np
