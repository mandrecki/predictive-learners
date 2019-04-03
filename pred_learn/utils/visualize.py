import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc


def states2video(record, filepath="0.avi"):
    width, height, _ = record[0]["s0"].shape
    FPS = 15

    fourcc = VideoWriter_fourcc(*'MP42')
    # fourcc = VideoWriter_fourcc(*'DIB ')
    video = VideoWriter(filepath, fourcc, float(FPS), (width, height))

    for timestep in record:
        video.write(timestep["s0"])

    video.release()
