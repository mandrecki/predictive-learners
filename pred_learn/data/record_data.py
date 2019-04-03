
import torch
import os

from pomme_players.managers import Zoo, Tournament

DATA_FOLDER = "datasets/first"


def add_more_data():
    # load a zoo
    z = Zoo()
    t = Tournament([], z.filler_players, z.ratings, n_repeat_player=12, parallel=True)
    print(t.standings)

    # start a tournament
    teams = t.pair()
    matches, idx2player = t.matchmake(teams)
    t.send_agents_to_envs(matches, idx2player)
    runs = t.envs.run_once_and_record()

    n_files_already = len(os.listdir(DATA_FOLDER))

    print(os.listdir(DATA_FOLDER))
    print(runs)

    runs = [run for run in runs if len(run) > 30]

    for i, run in enumerate(runs):
        filepath = '{0}/{1}.dict'.format(DATA_FOLDER, i + n_files_already)
        torch.save(run, open(filepath, 'wb'))

    t.envs.close()

if __name__ == "__main__":
    add_more_data()
