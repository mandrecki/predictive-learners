import numpy as np
import gym
from gym.spaces import Box


class AddShuffleNoise1D(gym.ObservationWrapper):
    def __init__(self, env=None, n_repeats=1, shuffle_once_per_reset=False):
        super(AddShuffleNoise1D, self).__init__(env)
        self.old_observation_shape = self.observation_space.shape[0]
        self.n_repeats = n_repeats
        self.reshuffle_once_per_reset = shuffle_once_per_reset
        self.shuffled_indices = [np.arange(self.observation_space.shape[0]) for _ in range(self.n_repeats)]
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [(n_repeats+1) * self.observation_space.shape[0]],
            dtype=self.observation_space.dtype)
        for i in range(self.n_repeats):
            np.random.shuffle(self.shuffled_indices[i])

    def observation(self, observation):
        new_obs = [observation]
        for i in range(self.n_repeats):
            if self.reshuffle_once_per_reset:
                obs_shuffled = observation[self.shuffled_indices[i]]
            else:
                obs_shuffled = np.copy(observation)
                np.random.shuffle(obs_shuffled)
            new_obs.append(obs_shuffled)
        new_obs_np = np.concatenate(new_obs)
        return new_obs_np

    def reset(self, **kwargs):
        for i in range(self.n_repeats):
            np.random.shuffle(self.shuffled_indices[i])
        return super(AddShuffleNoise1D, self).reset(**kwargs)


class AddResampledObs(gym.ObservationWrapper):
    def __init__(self, env=None, n_repeats=1, shuffle_once_per_reset=False):
        super(AddResampledObs, self).__init__(env)
        self.old_observation_shape = self.observation_space.shape[0]
        self.n_repeats = n_repeats
        self.reshuffle_once_per_reset = shuffle_once_per_reset
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [(n_repeats+1) * self.old_observation_shape],
            dtype=self.observation_space.dtype)
        self.shuffled_indices = (np.random.randint(0, self.old_observation_shape,
                                                   size=self.n_repeats * self.old_observation_shape))

    def observation(self, observation):
        new_obs = [observation]
        if not self.reshuffle_once_per_reset:
            self.shuffled_indices = (np.random.randint(0, self.old_observation_shape,
                                                       size=self.n_repeats * self.old_observation_shape))
        obs_shuffled = observation[self.shuffled_indices]
        new_obs.append(obs_shuffled)
        new_obs_np = np.concatenate(new_obs)
        return new_obs_np

    def reset(self, **kwargs):
        self.shuffled_indices = (np.random.randint(0, self.old_observation_shape,
                                                   size=self.n_repeats * self.old_observation_shape))
        return super(AddResampledObs, self).reset(**kwargs)


class AddBigNoise1D(gym.ObservationWrapper):
    def __init__(self, env=None, n_repeats=1, bigness=1):
        super(AddBigNoise1D, self).__init__(env)
        self.old_observation_shape = self.observation_space.shape[0]
        self.n_repeats = n_repeats
        self.bigness = bigness
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [(n_repeats+1) * self.observation_space.shape[0]],
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
        self.old_observation_shape = self.observation_space.shape[0]
        self.n_repeats = n_repeats
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [(n_repeats+1) * self.observation_space.shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        new_obs = [observation]
        for i in range(self.n_repeats):
            padding = np.zeros_like(observation)
            new_obs.append(padding)
        new_obs_np = np.concatenate(new_obs)
        return new_obs_np
