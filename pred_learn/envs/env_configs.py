ENVS = [
    "CarRacing-v0",
    "Snake-ple-v0",
    "Tetris-v0",
    "PuckWorld-ple-v0",
    "WaterWorld-ple-v0",
    "PixelCopter-ple-v0",
    "CubeCrash-v0",
    "Catcher-ple-v0",
    "Pong-ple-v0",
]

EXTRA_SMALL = 64
SMALL = 64
MEDIUM = 64

EXTRA_ARGS = {
    "Snake-ple-v0": {"width": EXTRA_SMALL, "height": EXTRA_SMALL, "init_length": 10},
    "PuckWorld-ple-v0": {"width": EXTRA_SMALL, "height": EXTRA_SMALL},
    "WaterWorld-ple-v0": {"width": EXTRA_SMALL, "height": EXTRA_SMALL},
    "PixelCopter-ple-v0": {"width": SMALL, "height": SMALL},
    # "CubeCrash-v0": {"width": EXTRA_SMALL, "height": EXTRA_SMALL},
    "Catcher-ple-v0": {"width": SMALL, "height": SMALL},
    "Pong-ple-v0": {"width": SMALL, "height": SMALL},
}