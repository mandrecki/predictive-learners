import matplotlib.pyplot as plt

from pred_learn.envs import *

if __name__ == "__main__":
    from pyvirtualdisplay import Display
    display = Display(visible=1, size=(600, 500))
    display.start()

    record = []
    seed = 0
    run_length = 25
    # for env_id in ALL_ENVS:
    # for env_id in GYM_ENVS:
    # for env_id in GAME_ENVS:
    for env_id in CONTROL_SUITE_ENVS:
        env = make_env(env_id, 0)
        obs = env.reset()
        for i in range(run_length):
            timestep = {}
            timestep["s0"] = np.copy(obs)
            action = env.sample_random_action()
            timestep["a0"] = np.copy(action)
            obs, rew, done, info = env.step(action)
            plt.figure(0)
            plt.clf()
            plt.imshow(obs)
            print("env:", env_id)
            print("reward:", rew)
            print("action:", action)
            plt.pause(0.02)

            timestep["r1"] = rew
            timestep["terminal"] = done
            # record.append(timestep)

            if done:
                break
        env.close()
    display.stop()
