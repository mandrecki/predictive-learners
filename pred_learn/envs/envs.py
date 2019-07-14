"""
Classes for configuring, standardising environments.

Wrapper types:
    * symbolic to image
    * lower image depth
    * output image bits (HxWxC -> HxWxCxbit_depth)
    * add shuffled channel
    * add noise channel
    * add image channel (unlearnable)
    * add video channel (learnable)

Requirements:

1. Conversions to image formats
2. Perturbations
3. Standardisation for action spaces (continuous and discrete)
4. Env vectorisation
5. Displaying episodes
6. Cropping for some envs (Tetris?)

Questions:

* Should environments return tensors or numpy arrays?
-> Numpy arrays for easy compatibility with model-free agents from other repos.

* Should bitrate dropping be done on env or agent side?
-> Agent, as some agents (model-free) may choose not to drop bits.

"""

import cv2
import numpy as np
from .wrappers import ToImageObservation, CropImage, ResizeImage, UnSuite

GAME_ENVS = [
    "CarRacing-v0",
    "TetrisA-v2",  # v2: reward score, penalize height
    "Snake-ple-v0",
    "PuckWorld-ple-v0",
    "WaterWorld-ple-v0",
    "PixelCopter-ple-v0",
    "CubeCrash-v0",
    "Catcher-ple-v0",
    "Pong-ple-v0",
]

GAME_ENVS_ACTION_REPEATS = {
    "TetrisA-v2" : 8,
    "CarRacing-v0": 4,
}


EXTRA_SMALL = 64
SMALL = 64
MEDIUM = 64
ENV_GAMES_ARGS = {
    "Snake-ple-v0": {"width": EXTRA_SMALL, "height": EXTRA_SMALL, "init_length": 10},
    "PuckWorld-ple-v0": {"width": EXTRA_SMALL, "height": EXTRA_SMALL},
    "WaterWorld-ple-v0": {"width": EXTRA_SMALL, "height": EXTRA_SMALL},
    "PixelCopter-ple-v0": {"width": SMALL, "height": SMALL},
    "Catcher-ple-v0": {"width": SMALL, "height": SMALL},
    "Pong-ple-v0": {"width": SMALL, "height": SMALL},
}


GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
            'Walker2d-v2']

GYM_ENVS_ACTION_REPEATS = {
    "CarRacing-v0": 6,
}

CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run',
                      'ball_in_cup-catch', 'walker-walk']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2}

ALL_ENVS = GAME_ENVS + GYM_ENVS + CONTROL_SUITE_ENVS

CROP_ENVS = {
    "TetrisA-v2": (45, 211, 93, 178),  # only blocks visible
}


def make_env(env_id, seed, max_episode_length):
    if env_id in GAME_ENVS:
        env = GameEnv(env_id, seed, max_episode_length=max_episode_length)
    elif env_id in GYM_ENVS:
        env = GymEnv(env_id, seed, max_episode_length=max_episode_length)
    elif env_id in CONTROL_SUITE_ENVS:
        env = ControlSuiteEnv(env_id, seed, max_episode_length=max_episode_length)
    else:
        raise ValueError("Bad env_id", env_id)

    # Crop and resize if necessary
    if env_id in CROP_ENVS.keys():
        env._env = CropImage(env._env, CROP_ENVS.get(env_id))
    if env.observation_size[0:2] != (64, 64):
        env._env = ResizeImage(env._env, (64, 64))

    return env


class GameEnv:
    def __init__(self, env_id, seed, max_episode_length=200):
        import gym
        extra_args = ENV_GAMES_ARGS.get(env_id, {})
        if env_id == "TetrisA-v2":
            import gym_tetris
            from gym_tetris.actions import SIMPLE_MOVEMENT
            from nes_py.wrappers import JoypadSpace
            # TODO resize and crop observation
            self._env = JoypadSpace(gym_tetris.make(env_id, **extra_args), SIMPLE_MOVEMENT)
        elif 'ple' in env_id:
            import gym_ple
            self._env = gym_ple.make(env_id, **extra_args)
        else:
            self._env = gym.make(env_id, **extra_args)

        self._env.seed(seed)
        self.action_repeat = GAME_ENVS_ACTION_REPEATS.get(env_id, 1)
        self.max_episode_length = max_episode_length * self.action_repeat
        self.t = 0

    def reset(self):
        self.t = 0  # Reset internal timer
        observation = self._env.reset()
        return observation

    def step(self, action):
        # action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            observation, reward_k, done, _ = self._env.step(action)
            # image = self._env.render(mode='rgb_array')
            # observation = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break

        return observation, reward, done, None

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self._env.action_space.sample()

    # TODO Add this and wrap full env?
    # @property
    # def action_space(self):
    #     return self._env.action_space
    #
    # @property
    # def observation_space(self):
    #     return self._env.observation_space


class GymEnv:
    def __init__(self, env_id, seed, max_episode_length=200):
        import gym
        self._env = ToImageObservation(gym.make(env_id))
        self._env.seed(seed)
        self.action_repeat = GYM_ENVS_ACTION_REPEATS.get(env_id, 1)
        self.max_episode_length = max_episode_length * self.action_repeat
        self.t = 0

    def reset(self):
        self.t = 0  # Reset internal timer
        observation = self._env.reset()
        return observation

    def step(self, action):
        reward = 0
        for k in range(self.action_repeat):
            observation, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        return observation, reward, done, None

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self._env.action_space.sample()


class ControlSuiteEnv:
    def __init__(self, env_id, seed, max_episode_length=200):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env_id.split('-')
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        self._env = pixels.Wrapper(self._env)
        self._env.action_space = self.action_size
        self._env.observation_space = self.observation_size
        self._env.reward_range = None
        self._env.metadata = None
        self._env = UnSuite(self._env)

        self.action_repeat = CONTROL_SUITE_ACTION_REPEATS.get(domain, 1)
        self.max_episode_length = max_episode_length * self.action_repeat
        if self.action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print('Using action repeat %d; recommended action repeat for domain is %d' % (
                self.action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
        self.t = 0

    def reset(self):
        self.t = 0  # Reset internal timer
        observation = self._env.reset()
        return observation

    def step(self, action):
        reward = 0
        for k in range(self.action_repeat):
            observation, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        return observation, reward, done, None

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_spec()['pixels'].shape

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)


# TODO update below

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    return observation // 2**(8-bit_depth) / 2**bit_depth
    # observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(
    #     0.5)  # Quantise to given bit depth and centre
    # observation.add_(torch.rand_like(observation).div_(
    #     2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(
        np.uint8)


def _images_to_observation(images, bit_depth):
    images = cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR)
    images = preprocess_observation_(images, bit_depth)
    return images


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return np.concatenate(np.expand_dims(observations, 0))

    # Steps/resets every environment and returns (observation, reward, done); returns blank observation once done
    def step(self, actions):
        observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        self.dones = dones
        return np.concatenate(np.expand_dims(observations, 0)), np.array(rewards), np.array(dones, dtype=np.uint8)

    def close(self):
        [env.close() for env in self.envs]
