import numpy as np
import gym
import cv2
import torch

from baselines.common.vec_env import VecEnvWrapper


# Standardising environments

class ToImageObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(ToImageObservation, self).__init__(env)
        image_size = (0, 0)
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*image_size, channels],
                                                dtype=np.uint8)

    def observation(self, symbolic_observation):
        image = self.render(mode='rgb_array')
        return image


class CropImage(gym.ObservationWrapper):
    def __init__(self, env, crop_box):
        super(CropImage, self).__init__(env)
        self.y_low, self.y_high, self.x_low, self.x_high = crop_box
        image_size = (self.y_high - self.y_low, self.x_high - self.x_low)
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*image_size, channels],
                                                dtype=np.uint8)

    def observation(self, image):
        image = image[self.y_low: self.y_high, self.x_low: self.x_high]
        return image


class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, new_size):
        super(ResizeImage, self).__init__(env)
        self.new_size = new_size
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*self.new_size, channels],
                                                dtype=np.uint8)

    def observation(self, image):
        # interpolation = cv2.INTER_LINEAR
        interpolation = cv2.INTER_AREA
        image = cv2.resize(image, self.new_size, interpolation=interpolation)
        return image


class UnSuite(gym.ObservationWrapper):
    def __init__(self, env):
        super(UnSuite, self).__init__(env)
        self._env = env

    def reset(self):
        state = self._env.reset()
        return self._env.physics.render(camera_id=0)

    def step(self, action):
        state = self._env.step(action)
        # observation = state.observation["pixels"]
        reward = state.reward
        done = state.last()
        return self._env.physics.render(camera_id=0), reward, done, None

    def observation(self, state):
        return self._env.physics.render(camera_id=0)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# Perturbing environments

# AddShuffledDims
# AddNoisedDims
# AddImages
# AddVideo






