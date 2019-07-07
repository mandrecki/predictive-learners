import numpy as np
import gym
import cv2


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
        image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_LINEAR)
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








