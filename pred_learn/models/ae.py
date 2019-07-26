"""
Predictive autoencoder: deterministic RNN for tansitions combined with convolutional encoding and decoding.
"""
import torch
from torch import nn


IM_CHANNELS = 3
# IM_WIDTH = 64

V_SIZE = 64
BS_SIZE = V_SIZE
# N_SIZE = 256
# D_SIZE = 64
# G_SIZE = 256

# EP_LEN = 100


def model_init(m, mean=0, std=0.01):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def full_weight_init(model):
    for m in model._modules:
        model_init(model._modules[m])


class Encoder(nn.Module):
    def __init__(self, im_channels=3, v_size=V_SIZE):
        super(Encoder, self).__init__()
        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1
            # size: (channels, 64, 64)
            nn.Conv2d(im_channels, 32, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            # size: (channels, 2, 2)
        )

        self.fc_seq = nn.Sequential(
            nn.Linear(256 * 2 * 2, v_size),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),

        )
        # self.apply(full_weight_init)

    def forward(self, x):
        h = self.conv_seq(x)
        out = self.fc_seq(h.view(h.size(0), -1))
        return out


class Decoder(nn.Module):
    def __init__(self, im_channels=3, bs_size=BS_SIZE):
        super(Decoder, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(bs_size, 1024),
            nn.ReLU(inplace=True),
        )

        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1 -- is this true for deconv?
            # size: (N_FILTERS, 2, 2)
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, im_channels, 6, stride=2),
            nn.Sigmoid(),
        )
        # self.apply(full_weight_init)

    def forward(self, x):
        h = self.fc_seq(x)
        out = self.conv_seq(h.view(x.size(0), 1024, 1, 1))
        return out


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, v_size=V_SIZE):
        super(ActionEncoder, self).__init__()
        self.action_dim = action_dim
        self.fc_seq = nn.Sequential(
            nn.Linear(self.action_dim, v_size),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
        )

    def forward(self, actions):
        batch_size = actions.size(0)
        a_onehot = torch.zeros((batch_size, self.action_dim), device=actions.device).float()
        a_onehot[range(batch_size), actions.view(-1)] = 1
        out = self.fc_seq(a_onehot)
        return out


class BeliefStatePropagator(nn.Module):
    def __init__(self, v_size=V_SIZE, bs_size=BS_SIZE):
        super(BeliefStatePropagator, self).__init__()
        self.gru = nn.GRU(v_size, bs_size, num_layers=1, batch_first=True)

    def forward(self, x, h=None):
        # if h is None:
        #     out = self.gru(x)
        # else:
        #     out = self.gru(x, h)

        out, h = self.gru(x, h)
        return out, h


class SimpleFF(nn.Module):
    def __init__(self, v_size=V_SIZE):
        super(SimpleFF, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(v_size, v_size),
            nn.ReLU(inplace=True),
            nn.Linear(v_size, v_size),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.fc_seq(x)
        return out


class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        h = self.enc(x)
        out = self.dec(h)
        return out
