import numpy as np
import torch
from cv2 import VideoWriter, VideoWriter_fourcc


def states2video(record, filepath="0.avi"):
    width, height, channels = record[0]["s0"].shape
    width = width if channels == 3 else 2*width
    FPS = 15

    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(filepath, fourcc, float(FPS), (width, height))

    for timestep in record:
        if channels == 3:
            video.write(timestep["s0"])
        elif channels == 6:
            im_wide = np.concatenate(np.split(timestep["s0"], 2, axis=2), axis=1)
            video.write(im_wide)
        else:
            raise ValueError("Bad image dim: {}".format(channels))

    video.release()

def visual_torch2numpy(torchim):
    """Tranpose channels to numpy convention (channel last)
    :param torchim: torch tensor (a, b, c, ...), C, H, W -> (a, b, c, ...), H, W, C
    """
    numpyim = torchim.tra


def stack2wideim(tensor):
    """ Converts stack of torch features to viewable image.

    :param tensor: (batch_size, channels * n_stack, 64, 64) torch tensor
    :return: batch_size * 64, n_stack * 64, 3 ready for displaying
    """
    assert tensor.dim() == 4
    n_splits = tensor.size(1)//3
    ims_y = []
    for i in range(tensor.size(0)):
        im_x = np.hstack(
            np.split(tensor[i, ...].cpu().numpy().transpose([1, 2, 0]).astype('uint8'), n_splits, axis=2))
        ims_y.append(im_x)

    im_display = np.concatenate(ims_y, axis=0)
    return im_display


def series2wideim(series, return_numpy=False, skip_detail=True):
    """Converts a series or batch of series of images to viewable image.

    :param series: batch_size, series_len (optional), channels, h, w
    :return:
    """
    # if batch given
    if series.dim() == 5:
        series = torch.cat([ser for ser in series], dim=-2)

    assert series.dim() == 4
    wide_im = torch.cat([t for t in series], dim=-1)

    if wide_im.size(-3) > 3:
        assert wide_im.size(-3) % 3 == 0
        if skip_detail:
            wide_im = wide_im[:3, ...]
        else:
            wide_im = torch.cat(torch.split(wide_im, 3, dim=-3), dim=-1)

    if return_numpy:
        wide_im = wide_im.detach().cpu().numpy().transpose([1, 2, 0])

    return wide_im


def append_losses(new_losses, losses_record=None):
    if losses_record is None:
        losses_record = {key: [] for key in new_losses.keys()}
    for key, value in new_losses.items():
        losses_record[key].append(value.item())
    return losses_record


def losses2numpy(losses_record):
    ars = [np.array(loss) for loss in losses_record.values()]
    ar = np.stack(ars, axis=-1)
    return ar