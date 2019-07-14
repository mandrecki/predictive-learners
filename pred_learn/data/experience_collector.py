import gym_ple
import gym
import gym_tetris
# import ple

import torch
import numpy as np
from skimage.transform import resize
import os
import argparse

from pred_learn.utils import states2video
from pred_learn.envs.envs import make_env


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gym recorder')
    parser.add_argument('--env-id', default='Pong-ple-v0',
                        help='env to record (see list in env_configs.py')
    parser.add_argument('--file-number', default=0, type=int)
    parser.add_argument('--ep-steps', default=4000,
                        help='maximum consecutive steps in env')
    parser.add_argument('--total-steps', default=2000, type=int,
                        help='total steps to record')

    args = parser.parse_args()

    record_dir = "recorded/{}/".format(args.env_id)
    record_path = "{}/{}.torch".format(record_dir, args.file_number)
    video_path = "{}/{}.avi".format(record_dir, args.file_number)

    try:
        os.makedirs(record_dir)
    except FileExistsError:
        pass

    # extra_args = ENV_GAMES_ARGS.get(args.env_id, {})
    # env = gym_ple.make(args.env_id, **extra_args)
    env = make_env(args.env_id, 0, max_episode_length=args.ep_steps)
    render = False
    record = []

    while len(record) < args.total_steps:
        obs = env.reset()
        while True:
            timestep = {}
            timestep["s0"] = np.copy(obs)
            action = env.sample_random_action()
            timestep["a0"] = action

            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            timestep["s1"] = np.copy(obs)
            timestep["r1"] = rew
            timestep["terminal"] = done
            record.append(timestep)

            if done:
                break

    torch.save(record, record_path)
    states2video(record, video_path)
