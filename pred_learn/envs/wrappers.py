import numpy as np
import gym
import cv2
import torch
from enum import IntEnum
import random

from baselines.common.vec_env import VecEnvWrapper, VecEnvObservationWrapper
from gym_minigrid.minigrid import MiniGridEnv
import mazenv


class MazeEnvImage(gym.ObservationWrapper):
    FIXED_SEED = 1337

    def __init__(self, env, randomize):
        super(MazeEnvImage, self).__init__(env)
        self.randomize = randomize
        self.maze_size = self.env.maze.shape
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*self.maze_size, channels],
                                                dtype=np.uint8)
        if not randomize:
            random.seed(self.FIXED_SEED)
            self.env = mazenv.Env(mazenv.prim(self.maze_size))

    def seed(self, seed=None):
        if self.randomize:
            random.seed(seed)
            self.env = mazenv.Env(mazenv.prim(self.maze_size))

    def reset(self):
        if self.randomize:
            self.env = mazenv.Env(mazenv.prim(self.maze_size))
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = reward / np.prod(self.maze_size)
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        board = np.zeros(self.observation_space.shape, dtype=np.uint8)
        board[..., 0] = 255 * obs[..., mazenv.env.WALL_CELL_FIELD]
        board[..., 1] = 255 * obs[..., mazenv.env.END_CELL_FIELD]
        board[..., 2] = 255 * obs[..., mazenv.env.CURRENT_CELL_FIELD]
        return board


class AbsoluteActionGrid(gym.ActionWrapper):
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        right = 0
        down = 1
        left = 2
        up = 3
        # Done completing task
        done = 4

    def __init__(self, env):
        super(AbsoluteActionGrid, self).__init__(env)
        self.actions = AbsoluteActionGrid.Actions

        # Actions are discrete integer values
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        if action < 4:
            # set agent's direction
            self.unwrapped.agent_dir = action
            action = MiniGridEnv.Actions.forward
        elif action == self.actions.done:
            action = MiniGridEnv.Actions.done
        else:
            raise ValueError("Bad action: {}".format(action))
        return action


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


class TinySokoban(gym.ObservationWrapper):
    def __init__(self, env):
        super(TinySokoban, self).__init__(env)
        image_size = (0, 0)
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*image_size, channels],
                                                dtype=np.uint8)

    def observation(self, symbolic_observation):
        image = self.render(mode='tiny_rgb_array')
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
    def __init__(self, env, new_size, antialias=False):
        super(ResizeImage, self).__init__(env)
        self.new_size = new_size
        self.antialias = antialias
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*self.new_size, channels],
                                                dtype=np.uint8)

    def add_padding(self, image):
        paddings = []
        for old_size in image.shape[0:2]:
            full_padding = 2 ** np.ceil(np.log2(old_size)) - old_size
            start_padding = int(np.floor(full_padding/2))
            end_padding = int(np.ceil(full_padding/2))
            paddings.append(start_padding)
            paddings.append(end_padding)

        if any(paddings):
            image = cv2.copyMakeBorder(image, *paddings, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image

    def observation(self, image):
        # only add padding if starting with small image
        if self.antialias and image.shape[0] != self.new_size[0]:
            image = self.add_padding(image)
            image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_NEAREST)
        else:
            image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_AREA)
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
        return self._env.physics.render(camera_id=0), reward, done, {}

    def observation(self, state):
        return self._env.physics.render(camera_id=0)


# Add detail
class ConcatNoise(gym.ObservationWrapper):
    def __init__(self, env, extra_channels=3):
        super(ConcatNoise, self).__init__(env)
        self.im_size = (64, 64)
        self.channels = extra_channels + 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*self.im_size, self.channels],
                                                dtype=np.uint8)

    def observation(self, observation):
        detail_image = np.random.randint(0, 255, (64, 64, 3))
        concat_image = np.concatenate((observation, detail_image), axis=2).astype(dtype=np.uint8)
        return concat_image


class ConcatRandomFrame(ConcatNoise):
    def __init__(self, env, im_file, extra_channels=3):
        super(ConcatRandomFrame, self).__init__(env)
        self.images = torch.load(im_file)

    def observation(self, observation):
        i = np.random.randint(0, self.images.shape[0])
        detail_image = self.images[i, ...]
        concat_image = np.concatenate((observation, detail_image), axis=2).astype(dtype=np.uint8)
        return concat_image


class ConcatVideo(ConcatRandomFrame):
    def __init__(self, env, im_file, extra_channels=3):
        super(ConcatVideo, self).__init__(env, im_file)
        self.i = None

    def reset(self):
        self.i = np.random.randint(0, self.images.shape[0])
        obs = self._env.reset()
        obs = self.observation(obs)
        return obs

    def observation(self, observation):
        detail_image = self.images[self.i, ...]
        self.i = (self.i + 1) % self.images.shape[0]
        concat_image = np.concatenate((observation, detail_image), axis=2).astype(dtype=np.uint8)
        return concat_image


class VecConcatVideo(VecEnvObservationWrapper):
    def __init__(self, venv, im_file, ordered=True):
        self.images = torch.load(im_file)
        self.data_len = self.images.shape[0]
        self.ordered = ordered
        self.indices = None

        self.venv = venv
        wos = venv.observation_space  # wrapped ob space
        observation_space = gym.spaces.Box(
            wos.low[0, 0, 0],
            wos.high[0, 0, 0],
            shape=[6, 64, 64],
            dtype=wos.dtype)

        super(VecConcatVideo, self).__init__(venv, observation_space=observation_space)

    # def observation(self, observation):
    #     new_ims = self.images[self.indices, ...].transpose(2, 0, 1)
    #     observation = np.concatenate([observation, new_ims], axis=1)
    #     if self.ordered:
    #         self.indices = (self.indices + 1) % self.data_len
    #     else:
    #         self.indices = np.random.randint(0, self.data_len, self.num_envs)
    #     return observation

    def process(self, observation):
        new_ims = self.images[self.indices, ...].transpose(0, 3, 1, 2)
        observation = np.concatenate([observation, new_ims], axis=1)
        if self.ordered:
            self.indices = (self.indices + 1) % self.data_len
        else:
            self.indices = np.random.randint(0, self.data_len, self.num_envs)
        return observation

    def reset(self):
        self.indices = np.random.randint(0, self.data_len, self.num_envs)
        obs = self.venv.reset()
        obs = self.process(obs)
        return obs


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






