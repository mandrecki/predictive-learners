import numpy as np
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


def series2wideim(tensor):
    """Converts a series or batch of series of images to viewable image.

    :param tensor: batch_size, series_len, channels, h, w
    :return:
    """
    assert tensor.dim() == 5
    # colour channel last
    x = tensor.cpu().numpy().transpose([0, 1, 3, 4, 2])
    n_splits = x.shape[-1]//3
    ims_y = []
    for i in range(x.shape[0]):
        im_x = np.dstack(np.split(x[i, ...], n_splits, axis=-1))
        im_x = np.dstack(np.split(im_x, im_x.shape[0], axis=0))
        ims_y.append(im_x)

    im_display = np.hstack(ims_y).squeeze(0)
    return im_display