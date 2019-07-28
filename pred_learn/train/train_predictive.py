import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse
import visdom

from pred_learn.models import AE_Predictor
from pred_learn.models.vae_wm import VAE
from pred_learn.data.data_container import ObservationSeriesDataset, ImageSeriesDataset
from pred_learn.utils import stack2wideim, series2wideim
from pred_learn.envs import make_env

IGNORE_N_FIRST_IN_LOSS = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gym recorder')
    parser.add_argument('--env-id', default='Pong-ple-v0',
                        help='env to record (see list in env_configs.py')
    parser.add_argument('--file-appendix', default="0", type=str)
    parser.add_argument('--rl-model-path', default=None, help='rl model to load for action selection')
    parser.add_argument('--extra-video', default=False, action='store_true', help='env with extra detail?')
    parser.add_argument('--extra-image', default=False, action='store_true', help='env with extra detail?')
    parser.add_argument('--video-path', default="../clean_records/test_vid.torch", help='path to ordered images')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')

    args = parser.parse_args()
    print("args given:", args)

    env_id = args.env_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tmp_env = make_env(args.env_id)
    action_space = tmp_env.action_space
    action_space_n = tmp_env.action_size
    series_len = 15
    batch_size = 16
    workers = 4

    dataset_train = ObservationSeriesDataset("../clean_records/{}/video-1.torch".format(env_id), action_space_n, series_len)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    # if device == "cuda:0":
    #     train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    # else:
    #     train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)

    # dataset_test = ObservationSeriesDataset("../clean_records/{}/video-1.torch".format(env_id), action_space_n, series_len)
    # test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=workers)

    channels_n = dataset_train.get_channels_n()
    model = AE_Predictor(channels_n, action_space_n).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss().to(device)
    losses = []

    if args.vis:
        vis = visdom.Visdom()
        win_target = None
        win_recon = None
        window_loss = None

    for i_epoch in range(20):
        for i_batch, batch in enumerate(train_loader):
            model.zero_grad()
            obs_in = batch["s0"].to(device)
            actions = batch["a0"].to(device)
            obs_target = batch["s1"].to(device)

            #         preprocess obs
            obs_in = obs_in.float() / 255
            obs_target = obs_target.float() / 255
    #
            obs_recon, obs_preds, _ = model.predict_full(obs_in, actions.long())
            loss = loss_fn(obs_preds[:, IGNORE_N_FIRST_IN_LOSS:, ...], obs_target[:, IGNORE_N_FIRST_IN_LOSS:, ...])
    #         #         loss = loss_fn(obs_recon, obs_in)
            losses.append(np.log(loss.item()))
            loss.backward()
            optimiser.step()

            if i_batch % 100 == 0 and args.vis:
                window_loss = vis.line(losses, win=window_loss)
                win_recon = vis.image(series2wideim(obs_recon), win=win_recon)
                win_target = vis.image(series2wideim(obs_target), win=win_target)
