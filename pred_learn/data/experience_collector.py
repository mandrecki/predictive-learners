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

from pred_learn.utils import states2video, stack2wideim
from pred_learn.envs.envs import make_rl_envs
from pred_learn.envs.envs import RL_SIZE, CHANNELS


P_NO_ACTION = 0.02

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gym recorder')
    parser.add_argument('--env-id', default='Pong-ple-v0',
                        help='env to record (see list in env_configs.py')
    parser.add_argument('--file-appendix', default="0", type=str)
    parser.add_argument('--n-envs', default=1, type=int)
    parser.add_argument('--rl-model-path', default=None, help='rl model to load for action selection')
    parser.add_argument('--render', default=False, action='store_true', help='render or not')
    parser.add_argument('--no-record', default=False, action='store_true', help='do not save records')
    parser.add_argument('--extra-video', default=False, action='store_true', help='env with extra detail?')
    parser.add_argument('--extra-image', default=False, action='store_true', help='env with extra detail?')
    parser.add_argument('--video-path', default="../clean_records/test_vid.torch", help='path to ordered images')
    parser.add_argument('--total-steps', default=2000, type=int,
                        help='total steps to record')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')

    args = parser.parse_args()
    print("args given:", args)

    record_dir = "recorded/{}/".format(args.env_id)
    record_path = "{}/{}.torch".format(record_dir, args.file_appendix)
    video_path = "{}/{}.avi".format(record_dir, args.file_appendix)

    assert not (args.extra_video and args.extra_image)
    extra_detail = args.extra_video or args.extra_image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame_stack = 1 if args.recurrent_policy else 4
    # device = "cpu"
    envs = make_rl_envs(args.env_id, seed=np.random.randint(0, 10000), n_envs=args.n_envs,
                        device=device,
                        frame_stack=frame_stack,
                        add_video=args.extra_video, add_frames=args.extra_image,
                        vid_path=args.video_path)

    channels = envs.observation_space.shape[0] // frame_stack

    if args.rl_model_path is not None:
        actor, _ = torch.load(args.rl_model_path)
        actor = actor.to(device)
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
    else:
        plt.figure(1)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    while len(record) < args.total_steps:
        obs = envs.reset()
        im = obs[0, ...].cpu().numpy().transpose([1, 2, 0])[:, :, -channels:].astype("uint8")
        if actor:
            rnn_hxs = None
        while len(record) < args.total_steps:
            if len(record) % 1000 == 0:
                print("Timesteps so far...", len(record))

            timestep = {}
            timestep["s0"] = np.copy(im)
            if actor:
                with torch.no_grad():
                    _, action, _, rnn_hxs = actor.act(obs, rnn_hxs, None)

                    if args.render:
                        plt.figure(1)
                        plt.clf()
                        im_display = stack2wideim(obs)
                        plt.imshow(im_display)
                        envs.render()
                        plt.pause(0.05)

            else:
                action = envs.sample_random_action()

            # set action to null action
            if np.random.rand() < P_NO_ACTION:
                action[:] = 0

            # if args.render:
            #     env.render()
            #     plt.pause(0.02)

            timestep["a0"] = action[0, ...].cpu().numpy()
            obs, rew, done, info = envs.step(action)
            im = obs[0, ...].cpu().numpy().transpose([1, 2, 0])[:, :, -channels:].astype("uint8")

            timestep["s1"] = np.copy(im)
            timestep["r1"] = rew[0, ...].cpu().numpy()
            timestep["terminal"] = done[0, ...].item()
            record.append(timestep)
            # if not args.no_record:
            #     pass

            # if args.render:
            #     print("Action:", action)
            #     print("Reward:", rew)

            if done[0, ...]:
                break

    if not args.no_record:
        torch.save(record, record_path)
        states2video(record, video_path)

    envs.close()
