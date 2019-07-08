import torch
from torch import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Resize, ToTensor, ToPILImage, Grayscale, RandomGrayscale
import os
import numpy as np
from collections import deque
from skimage.transform import resize



ALPHA = 0.6
BETA = 0.5


class ObservationDataset(Dataset):
    def __init__(self, filepath, action_space_n):
        self.record = torch.load(filepath)
        self.action_space_n = action_space_n

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        # idx = np.random.randint(0, len(self))

        timestep = self.record[idx]
        action = timestep["a0"]
        if type(action) is int:
            action = np.zeros(self.action_space_n)
            action[timestep["a0"]] = 1

        sample = {
            "s0": torch.FloatTensor(timestep["s0"]/255).permute([2, 1, 0]),
            "s1": torch.FloatTensor(timestep["s1"]/255).permute([2, 1, 0]),
            "a0": torch.LongTensor(action),
            "r1": torch.FloatTensor([timestep["r1"]]),
            "terminal": torch.ByteTensor([timestep["terminal"]]),
        }

        return sample

    def extend(self, iterable):
        self.record.extend(iterable)


class ObservationSeriesDataset(Dataset):
    def __init__(self, filepath, action_space_n, series_length):
        self.record = torch.load(filepath)
        self.action_space_n = action_space_n
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
            sample["s0"].append(torch.FloatTensor(timestep["s0"]/255).permute([2, 1, 0]).unsqueeze(0))
            sample["s1"].append(torch.FloatTensor(timestep["s1"]/255).permute([2, 1, 0]).unsqueeze(0))
            sample["a0"].append(torch.LongTensor([timestep["a0"]]).unsqueeze(0))
            sample["r1"].append(torch.FloatTensor([timestep["r1"]]).unsqueeze(0))
            sample["terminal"].append(torch.ByteTensor([timestep["terminal"]]).unsqueeze(0))

        for key, value in sample.items():
            sample[key] = torch.cat(value)

        return sample

    def extend(self, iterable):
        self.record.extend(iterable)


class ReplayBuffer(Dataset):
    def __init__(self, buff_len, reward_function):
        self.deque = deque(maxlen=buff_len)
        self.reward_func = reward_function

    def __len__(self):
        return len(self.deque)

    def __getitem__(self, _):
        idx = np.random.randint(0, len(self))

        data_t = self.deque[idx]
        reward = self.reward_func(data_t["game_delta"])

        s0 = {key: torch.FloatTensor(feature) for key, feature in data_t["s0"].items()}
        s1 = {key: torch.FloatTensor(feature) for key, feature in data_t["s1"].items()}

        sample = {
            "s0": s0,
            "a0": torch.Tensor([data_t["a0"]]),
            "s1": s1,
            # "r1": torch.Tensor([data_t["r1"]]),
            "r1": torch.Tensor([reward]),
            "terminal": torch.ByteTensor([data_t["terminal"]]),
        }

        return sample, idx, 1/len(self)

    def extend(self, iterable):
        self.deque.extend(iterable)


class SeriesReplayBuffer(Dataset):
    def __init__(self, buff_len):
        self.deque = deque(maxlen=buff_len)
        self.series_length = 1

    def __len__(self):
        return len(self.deque)

    def __getitem__(self, idx):
        idx = min(idx, len(self) - self.series_length - 1)

        sample = {
            "s0": [],
            "a0": [],
            "s1": [],
            # "r1": torch.Tensor([data_t["r1"]]),
            "game_delta": [],
            "terminal": [],
        }
        for i in range(idx, idx+self.series_length):
            data_t = self.deque[i]

            s0 = {key: torch.FloatTensor(feature) for key, feature in data_t["s0"].items()}
            s1 = {key: torch.FloatTensor(feature) for key, feature in data_t["s1"].items()}
            game_delta = list(data_t["game_delta"].values())
            sample["s0"].append(s0)
            sample["s1"].append(s1)
            sample["a0"].append(torch.LongTensor([data_t["a0"]]))
            sample["game_delta"].append(torch.Tensor(game_delta))
            sample["terminal"].append(torch.ByteTensor([data_t["terminal"]]))

        return sample, idx, 1/len(self)

    def extend(self, iterable):
        self.deque.extend(iterable)



class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buff_len, tensorize_single, reward_function, alpha=ALPHA):
        super(PrioritizedReplayBuffer, self).__init__(buff_len, tensorize_single, reward_function)
        self.priorities = deque(maxlen=buff_len)
        self.probabilities = None

        self.alpha = alpha
        # self.beta = BETA

    def __getitem__(self, idx):
        data_t = self.deque[idx]
        p = self.probabilities[idx]

        reward = self.reward_func(data_t["game_delta"])

        s0 = {key: torch.FloatTensor(feature) for key, feature in data_t["s0"].items()}
        s1 = {key: torch.FloatTensor(feature) for key, feature in data_t["s1"].items()}

        sample = {
            "s0": s0,
            "a0": torch.Tensor([data_t["a0"]]),
            "s1": s1,
            # "r1": torch.Tensor([data_t["r1"]]),
            "r1": torch.Tensor([reward]),
            "terminal": torch.ByteTensor([data_t["terminal"]]),
        }

        # sample = {
        #     "s0": self.tensorize(data_t["s0"]),
        #     "a0": torch.Tensor([data_t["a0"]]),
        #     "s1": self.tensorize(data_t["s1"]),
        #     # "r1": torch.Tensor([data_t["r1"]]),
        #     "r1": torch.Tensor([reward]),
        #     "terminal": torch.ByteTensor([data_t["terminal"]]),
        # }

        return sample, idx, p

    def extend(self, iterable):
        self.deque.extend(iterable)
        max_priority = max(self.priorities) if len(self.priorities) > 0 else 10.0
        new_priorities = [max_priority] * len(iterable)
        self.priorities.extend(new_priorities)
        assert len(self.deque) == len(self.priorities)
        self.update_probabilities()

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            # print(i.item(), len(self.priorities), self.priorities[i.item()])
            self.priorities[i.item()] = (priority + 1e-5) ** self.alpha
        self.update_probabilities()

    def update_probabilities(self):
        self.probabilities = np.array(self.priorities)
        self.probabilities /= np.sum(self.probabilities)


class ReplayBufferSampler(Sampler):
    def __init__(self, replay_buffer):
        self.data_source = replay_buffer

    def __iter__(self):
        while True:
            yield np.random.randint(0, len(self.data_source))

    def __len__(self):
        return np.Inf


class PrioritisedReplaySampler(Sampler):
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer

    def __iter__(self):
        while True:
            yield np.random.choice(len(self.buffer.probabilities), p=self.buffer.probabilities)

    def __len__(self):
        return np.Inf


class DummySampler(Sampler):
    def __init__(self):
        pass

    def __iter__(self):
        while True:
            yield 0

    def __len__(self):
        return np.Inf

if __name__ == "__main__":
    data_loader = DataLoader(1000, batch_size=4, shuffle=True, num_workers=1)

    for i, batch in zip(range(3), data_loader):
        print("Batch i", i)
        print(batch)