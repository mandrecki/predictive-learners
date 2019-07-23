# import gym_ple
# import gym
# import gym_tetris
# import ple

import torch
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from skimage.transform import resize
import os
import argparse

from baselines import bench

from pred_learn.utils import states2video
from pred_learn.envs.envs import make_env
from pred_learn.envs.envs import RL_SIZE, CHANNELS


P_NO_ACTION = 0.1

class ObsBuffer:
    def __init__(self, max_len=4, channels=CHANNELS):
        self.max_len = max_len
        self.channels = channels
        self.buffer = np.zeros((1, self.max_len * self.channels, RL_SIZE, RL_SIZE))

    def reset(self):
        self.buffer[:] = 0

    def add_obs(self, observation):
        self.buffer[:, :-self.channels, ...] = self.buffer[:, self.channels:, ...]
        self.buffer[:, -self.channels:, ...] = observation.transpose([2, 0, 1])
        # self.buffer = np.roll(self.buffer, -self.channels, axis=1)
        # self.buffer[0, -self.channels:, ...] = observation.transpose([2, 0, 1])
        # self.buffer = np.roll(self.buffer, self.channels, axis=1)
        # self.buffer[0, 0:self.channels, ...] = observation.transpose([2, 0, 1])

    def get_tensor(self):
        return torch.Tensor(self.buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gym recorder')
    parser.add_argument('--env-id', default='Pong-ple-v0',
                        help='env to record (see list in env_configs.py')
    parser.add_argument('--file-number', default=0, type=int)
    parser.add_argument('--rl-model-path', default=None, help='rl model to load for action selection')
    parser.add_argument('--render', default=False, action='store_true', help='render or not')
    parser.add_argument('--extra-detail', default=False, action='store_true', help='env with extra detail?')
    parser.add_argument('--ep-steps', default=4000,
                        help='maximum consecutive steps in env')
    parser.add_argument('--total-steps', default=2000, type=int,
                        help='total steps to record')

    args = parser.parse_args()
    print("args given:", args)

    record_dir = "recorded/{}/".format(args.env_id)
    record_path = "{}/{}.torch".format(record_dir, args.file_number)
    video_path = "{}/{}.avi".format(record_dir, args.file_number)

    extra_detail = args.extra_detail
    env = make_env(args.env_id, np.random.randint(0, 10000), max_episode_length=args.ep_steps, extra_detail=extra_detail)

    if args.rl_model_path is not None:
        actor, _ = torch.load(args.rl_model_path)
    else:
        actor = None
        print("No RL model provided. Taking random actions.")

    try:
        os.makedirs(record_dir)
    except FileExistsError:
        pass

    record = []

    if not args.render:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(600, 500))
        display.start()

    while len(record) < args.total_steps:
        if len(record) % 1000 == 0:
            print("Timestops so far...", len(record))

        obs = env.reset()
        if actor:
            buffer = ObsBuffer(channels=obs.shape[2])
            rnn_hxs = None
            buffer.reset()
        while len(record) < args.total_steps:
            timestep = {}
            timestep["s0"] = np.copy(obs)
            if actor:
                with torch.no_grad():
                    buffer.add_obs(obs)
                    obs_tensor = buffer.get_tensor()
                    # im_display = obs_tensor[0, -CHANNELS:, ...].numpy().transpose([1, 2, 0]).astype('uint8')
                    n_splits = 8 if extra_detail else 4
                    im_display = np.concatenate(np.split(obs_tensor[0, ...].numpy().transpose([1, 2, 0]).astype('uint8'), n_splits, axis=2), axis=1)
                    _, action, _, rnn_hxs = actor.act(obs_tensor, rnn_hxs, None)
                    action = action.numpy()[0]
                    if len(action) == 1:
                        action = action[0]

                    if args.render:
                        plt.figure(1)
                        plt.clf()
                        plt.imshow(im_display)
                        plt.pause(0.05)


            else:
                action = env.sample_random_action()

            # set action to null action
            if np.random.rand() < P_NO_ACTION:
                if np.isscalar(action):
                    action = 0
                else:
                    action[:] = 0

            # if args.render:
            #     env.render()
            #     plt.pause(0.02)

            timestep["a0"] = action
            obs, rew, done, info = env.step(action)

            timestep["s1"] = np.copy(obs)
            timestep["r1"] = rew
            timestep["terminal"] = done
            record.append(timestep)

            if args.render:
                print("Action:", action)
                print("Reward:", rew)

            if done:
                break

    torch.save(record, record_path)
    states2video(record, video_path)
