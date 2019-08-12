import torch
from torch import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Resize, ToTensor, ToPILImage, Grayscale, RandomGrayscale
import os
import numpy as np
from collections import deque
from skimage.transform import resize


class ObservationDataset(Dataset):
    def __init__(self, filepath, action_space_n, bit_depth=8):
        super(ObservationDataset, self).__init__()
        self.record = torch.load(filepath)
        self.action_space_n = action_space_n

        assert type(bit_depth) is int and 1 <= bit_depth <= 8
        self.bit_depth = bit_depth

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        # idx = np.random.randint(0, len(self))

        timestep = self.record[idx]
        action = timestep["a0"]
        if type(action) is int:
            action = np.zeros(self.action_space_n)
            action[timestep["a0"]] = 1

        s0 = self.preprocess_observation(timestep["s0"])
        s1 = self.preprocess_observation(timestep["s1"])

        sample = {
            "s0": torch.FloatTensor(s0).permute([2, 0, 1]),
            "s1": torch.FloatTensor(s1).permute([2, 0, 1]),
            "a0": torch.LongTensor(action),
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


class ObservationSeriesDataset(ObservationDataset):
    def __init__(self, filepath, action_space_n, series_length, bit_depth=8):
        super(ObservationSeriesDataset, self).__init__(filepath, action_space_n, bit_depth)
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

            sample["s0"].append(torch.FloatTensor(s0).permute([2, 0, 1]).unsqueeze(0))
            sample["s1"].append(torch.FloatTensor(s1).permute([2, 0, 1]).unsqueeze(0))
            sample["a0"].append(torch.Tensor([timestep["a0"]]).unsqueeze(0))
            sample["r1"].append(torch.FloatTensor([timestep["r1"]]).unsqueeze(0))
            # TODO remove item() below
            sample["terminal"].append(torch.ByteTensor([timestep["terminal"]]).unsqueeze(0))

        for key, value in sample.items():
            sample[key] = torch.cat(value)

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
