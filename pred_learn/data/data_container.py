import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gym


class ObservationDataset(Dataset):
    def __init__(self, filepath, action_space, bit_depth=8):
        super(ObservationDataset, self).__init__()
        self.record = torch.load(filepath)
        self.action_space = action_space
        if type(action_space) is gym.spaces.discrete.Discrete:
            self.action_size = action_space.n
        elif type(action_space) is gym.spaces.Box:
            self.action_size = action_space.shape[0]
        elif type(action_space) is int:
            self.action_size = action_space
        else:
            raise ValueError("Bad action space type given to dataset: {}".format( type(action_space)))

        assert type(bit_depth) is int and 1 <= bit_depth <= 8
        self.bit_depth = bit_depth

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        # idx = np.random.randint(0, len(self))

        timestep = self.record[idx]

        s0 = self.preprocess_observation(timestep["s0"])
        s1 = self.preprocess_observation(timestep["s1"])
        a0 = self.preprocess_action(timestep["a0"])

        sample = {
            "s0": torch.FloatTensor(s0).permute([2, 0, 1]),
            "s1": torch.FloatTensor(s1).permute([2, 0, 1]),
            "a0": torch.Tensor(a0),
            "r1": torch.FloatTensor([timestep["r1"]]),
            "terminal": torch.ByteTensor([timestep["terminal"]]),
        }

        return sample

    def extend(self, iterable):
        self.record.extend(iterable)

    def get_channels_n(self):
        return self.record[0]["s0"].shape[-1]

    def preprocess_observation(self, observation):
        # drop bits and to float
        return observation // 2 ** (8 - self.bit_depth) / 2 ** self.bit_depth

    def preprocess_action(self, action):
        if type(self.action_space) is gym.spaces.Discrete:
            action = np.eye(self.action_size)[action]
            action = action.squeeze()
            assert len(action.shape) == 1
        return action


class ObservationSeriesDataset(ObservationDataset):
    def __init__(self, filepath, action_space, series_length, bit_depth=8):
        super(ObservationSeriesDataset, self).__init__(filepath, action_space, bit_depth)
        self.series_length = series_length

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        idx = min(idx, len(self) - self.series_length - 1)

        sample = {
            "s0": [],
            "a0": [],
            "s1": [],
            "r1": [],
            "terminal": [],
        }
        for i in range(idx, idx+self.series_length):
            timestep = self.record[i]
            s0 = self.preprocess_observation(timestep["s0"])
            s1 = self.preprocess_observation(timestep["s1"])
            a0 = self.preprocess_action(timestep["a0"])

            sample["s0"].append(torch.FloatTensor(s0).permute([2, 0, 1]))
            sample["s1"].append(torch.FloatTensor(s1).permute([2, 0, 1]))
            sample["a0"].append(torch.Tensor(a0))
            sample["r1"].append(torch.FloatTensor(timestep["r1"]))
            sample["terminal"].append(torch.ByteTensor([timestep["terminal"]]))

        for key, value in sample.items():
            sample[key] = torch.stack(value)

        return sample

    def extend(self, iterable):
        self.record.extend(iterable)


class ImageDataset(Dataset):
    def __init__(self,  filepath):
        super(ImageDataset, self).__init__()
        self.images = torch.load(filepath)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        im = self.images[idx, ...]
        sample = torch.Tensor(im).permute([2, 0, 1])
        return sample


class ImageSeriesDataset(ImageDataset):
    def __init__(self, filepath, series_length, ordered=True):
        super(ImageSeriesDataset, self).__init__(filepath)
        self.series_length = series_length
        self.ordered = ordered

    def __getitem__(self, idx):
        if self.ordered:
            idx = min(idx, len(self) - self.series_length - 1)
            im = self.images[idx:idx+self.series_length, ...]
        else:
            im = self.images[np.random.randint(0, len(self), self.series_length), ...]

        sample = torch.Tensor(im).permute([0, 3, 1, 2])
        return sample
