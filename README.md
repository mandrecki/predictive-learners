# Predictive learners (pred_learn package)

For evaluating agents that learn predictive models and then use those in RL setting.

## Installation

Lots of requirements, might streamline this later.
Tested on Python3.6.

Then clone this repo, install requirements, install this package:

```
git clone https://github.com/EqThinker/predictive-learners.git
pip install tensorflow # (consider GPU supporting version)
cd predictive-learners
pip install -r requirements.txt
pip install -e .
```

### Requirements

* pytorch
* numpy
* https://github.com/EqThinker/pytorch-a2c-ppo-acktr (for RL)
* pandas
* opencv-python
* pyvirtualdisplay
* more see requirements.txt

Gyms (install only those that you're planning to run)
* gym
* gym_ple
* gym_tetris
* mujoco
* deepmind suite

## Environments available
* gym classics: CartPole, MountainCar, etc.
* games: CarRacing, PixelCopter, Tetris, etc.
* Deepmind control suite: 


## Running experiments

Use files in scripts as starting point
