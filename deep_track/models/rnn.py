import torch
from torch import nn


class PredictorRNN(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size=16):
        super(PredictorRNN, self).__init__()
        self.rnn = nn.GRU(obs_shape + action_shape, hidden_size, num_layers=1, batch_first=True)

        self.mlp_us = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_shape),
        )

    def forward(self, obs, actions):
        h = torch.cat((obs, actions), dim=2)
        h, state_f = self.rnn(h)
        obs_shifted = self.mlp_us(h)
        return obs_shifted

        # net = PredictorRNN()
        # x = torch.zeros(150, 16, 4)
        # out = net(x)
        # out.size()
