import torch
from torch import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, ToPILImage, Grayscale, RandomGrayscale
import os
import numpy as np

GUARANTEED_PERCEPTS = 4

DATA_FOLDER = "datasets/first"


class GameStateSeqToTensor(object):
    def __init__(self):
        pass
        # self.to_grey = Grayscale()
        # self.resize = Resize(size)

    def __call__(self, state_seq):
        tensor_seq = [torch.from_numpy(state["states"][0]["board"]) for state in state_seq]
        return torch.stack(tensor_seq)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# class TensorToImSeq(object):
#     def __init__(self, size=(190, 210)):
#         self.to_im = ToPILImage()
#         self.resize = Resize(size)
#
#     def __call__(self, tensor):
#         tensor_seq = F.unbind(tensor, 1)
#         im_seq = [self.resize(self.to_im(tensor)) for tensor in tensor_seq]
#         return im_seq
#
#     def __repr__(self):
#         return self.__class__.__name__ + '()'


class PommeGamesDataset(Dataset):
    def __init__(self, root_dir, sequence_len=10, subtract_mean=False, p=1.0):
        self.root_dir = root_dir
        self.file_list = os.listdir(self.root_dir)
        self.tensorize = GameStateSeqToTensor()

        self.sequence_len = sequence_len
        # self.av_screen = None
        # self.subtract_mean = subtract_mean
        # self.p = p
        # if self.subtract_mean:
        #     self.av_screen = self.compute_av_screen()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = '{}/{}'.format(self.root_dir, self.file_list[idx])
        run = torch.load(filepath)

        # print(run)
        # print(len(run))
        t_start = np.random.randint(0, len(run)-self.sequence_len)
        run = run[t_start: t_start+self.sequence_len]
        return self.tensorize(run)

        # run['screens'] = self.tensorize(run['screens'])
        # if self.subtract_mean:
        #     assert self.av_screen is not None
        #     run['screens'] -= self.av_screen
        #
        # run['masked_screens'] = self.mask_episode(run['screens'])

    # def compute_av_screen(self):
    #     ep_mean = None
    #     data_len = len(self)
    #     for idx in range(data_len):
    #         filepath = '{}/{}.dict'.format(self.root_dir, idx)
    #         sample = torch.load(filepath)
    #         sample['screens'] = self.tensorize(sample['screens'])
    #         mean over episode
            # if ep_mean is None:
            #     ep_mean = torch.mean(sample['screens'], dim=0)
            # else:
            #     ep_mean += torch.mean(sample['screens'], dim=0)
        # return ep_mean/data_len
    #
    # def set_p(self, p):
    #     self.p = p
    #
    # def mask_episode(self, ep, return_indices=False):
    #     ep_len = ep.size(0)
    #     masked_ep = ep.clone()
    #     if self.p < 1.0:
    #         for_removal = torch.rand(ep_len) < self.p
    #     else:
    #         for_removal = torch.ones(ep_len) > 0
    #
    #     for_removal[0:GUARANTEED_PERCEPTS] = 0
    #     try:
    #         masked_ep[for_removal.nonzero(), ...] = 0
    #     except IndexError:
    #         print("Failed in masking", for_removal.nonzero())
            # return masked_ep
        #
        # return masked_ep

if __name__ == "__main__":
    data = PommeGamesDataset(DATA_FOLDER)
    data_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=1)

    for i, batch in zip(range(3), data_loader):
        print("Batch i", i)
        print(batch)
