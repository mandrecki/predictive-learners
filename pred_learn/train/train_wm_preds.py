import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse
import visdom

from pred_learn.models.predictors import VAE_MDN
from pred_learn.models.vae_wm import VAE
from pred_learn.data.data_container import ObservationSeriesDataset, ImageSeriesDataset, ObservationDataset
from pred_learn.utils.visualize import stack2wideim, series2wideim, losses2numpy, append_losses
from pred_learn.envs import make_env

IGNORE_N_FIRST_IN_LOSS = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gym recorder')
    parser.add_argument('--env-id', default='Pong-ple-v0',
                        help='env to record (see list in env_configs.py')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--bit-depth', type=int, default=8)
    parser.add_argument('--series-length', type=int, default=50)

    parser.add_argument('--file-appendix', default="0", type=str)
    parser.add_argument('--model-path', default=None, help='model to load if exists, then save to this location')
    parser.add_argument('--extra-video', default=False, action='store_true', help='env with extra detail?')
    parser.add_argument('--extra-image', default=False, action='store_true', help='env with extra detail?')
    parser.add_argument('--video-path', default="../clean_records/test_vid.torch", help='path to ordered images')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log interval, one log per n updates (default: 100)')

    args = parser.parse_args()
    print("args given:", args)

    env_id = args.env_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tmp_env = make_env(args.env_id)
    action_space = tmp_env.action_space
    action_space_n = tmp_env.action_size
    series_len = 15
    batch_size = args.batch_size
    workers = 4

    dataset_train = ObservationSeriesDataset("../clean_records/{}/base-1.torch".format(env_id), action_space_n, args.series_length, args.bit_depth)
    dataset_test = ObservationSeriesDataset("../clean_records/{}/base-2.torch".format(env_id), action_space_n, args.series_length, args.bit_depth)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    tmp_env.close()
    tmp_env = None

    channels_n = dataset_train.get_channels_n()
    model = VAE_MDN(channels_n, action_space_n).to(device)
    try:
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path))
    except FileNotFoundError:
        print("Model not found")

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = None
    test_losses = None

    if args.vis:
        vis = visdom.Visdom(env=env_id)
        win_target = None
        win_recon = None
        win_freerun = None
        window_loss = None
        window_test_loss = None

    for i_epoch in range(args.n_epochs):
        for i_batch, batch in enumerate(train_loader):
            model.zero_grad()
            obs_in = batch["s0"].to(device)
            obs_target = batch["s1"].to(device)
            actions = batch["a0"].to(device)
            rewards = batch["r1"].to(device)
            dones = batch["terminal"].to(device)

            loss, obs_pred = model.get_prediction_loss(obs_in, obs_target, actions, rewards, dones, return_recons=False)
            losses = append_losses(loss, losses)
            loss["total"].backward()
            optimiser.step()

            if i_batch % 100 == 0:
                with torch.no_grad():
                    batch = next(iter(test_loader))
                    obs_in = batch["s0"].to(device)
                    actions = batch["a0"].to(device)
                    obs_target = batch["s1"].to(device)
                    rewards = batch["r1"].to(device)
                    dones = batch["terminal"].to(device)

                    loss, obs_pred = model.get_prediction_loss(obs_in, obs_target, actions, rewards, dones,
                                                               return_recons=True)
                    freerun_pred = model.free_running_prediction(obs_in, actions, deterministic=True)
                    test_losses = append_losses(loss, test_losses)

                    if args.vis:
                        window_test_loss = vis.line(losses2numpy(test_losses), args.log_interval * np.arange(len(test_losses["total"])),
                                                    win=window_test_loss,
                                                    opts=dict(legend=list(test_losses.keys()), title="test loss"))
                        window_loss = vis.line(losses2numpy(losses), np.arange(len(losses["total"])),
                                               win=window_loss,
                                               opts=dict(legend=list(losses.keys()), title="training loss"))
                        win_recon = vis.image(series2wideim(obs_pred), win=win_recon, opts=dict(caption="preds"))
                        win_target = vis.image(series2wideim(obs_target), win=win_target, opts=dict(caption="target"))
                        win_freerun = vis.image(series2wideim(freerun_pred), win=win_freerun, opts=dict(caption="freerun"))

        if i_epoch % 1 == 0:
            torch.save(model.state_dict(), args.model_path)