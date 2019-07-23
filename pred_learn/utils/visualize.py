import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc


def states2video(record, filepath="0.avi"):
    width, height, channels = record[0]["s0"].shape
    FPS = 15

    fourcc = VideoWriter_fourcc(*'MP42')
    # fourcc = VideoWriter_fourcc(*'DIB ')
    video = VideoWriter(filepath, fourcc, float(FPS), (width, height))

    for timestep in record:
        # print(timestep["s0"][:, :, 0:3].shape)
        video.write(timestep["s0"][:, :, 0:3])

    video.release()
