import torch
from torch import nn


class ConvTranistion(nn.Module):
    def __init__(self, in_shape, n1, n2, n3):
        super(ConvTranistion, self).__init__()

        self.in_shape = in_shape
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.maxpool = nn.MaxPool2d(kernel_size=in_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n1, n1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1 + n2, n3, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.pool_and_inject(inputs)
        x = torch.cat([self.conv1(x), self.conv2(x)], 1)
        x = self.conv3(x)
        x = torch.cat([x, inputs], 1)
        return x

    def pool_and_inject(self, x):
        pooled = self.maxpool(x)
        tiled = pooled.expand((x.size(0),) + self.in_shape)
        out = torch.cat([tiled, x], 1)
        return out


class RecursiveConvTransition(nn.Module):
    def __init__(self, board_shape, num_actions, num_rewards, memory_layers):
        super(RecursiveConvTransition, self).__init__()

        self.board_channels = board_shape[0]
        self.width = board_shape[1]
        self.height = board_shape[2]
        self.num_actions = num_actions
        self.memory_layers = memory_layers

        self.conv = nn.Sequential(
            nn.Conv2d(self.board_channels + self.num_actions + self.memory_layers, 64, kernel_size=1),
            nn.ReLU()
        )

        self.basic_block1 = ConvTranistion((64, self.width, self.height), 16, 32, 64)
        self.basic_block2 = ConvTranistion((128, self.width, self.height), 16, 32, 64)

        self.image_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.ReLU()
        )
        self.image_gen = nn.Conv2d(256, self.board_channels, kernel_size=1)
        self.mem_gen = nn.Conv2d(256, self.memory_layers, kernel_size=1)

        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reward_fc = nn.Linear(64 * self.width * self.height, num_rewards)

    def forward(self, board, action, memory=None, action_mask=None):
        batch_size = board.size(0)

        if memory is None:
            memory = torch.zeros(batch_size, self.memory_layers, self.width, self.height)
            memory = memory.to(board.device)

        onehot_actions = torch.zeros(batch_size, self.num_actions, self.width, self.height)
        onehot_actions = onehot_actions.to(board.device)
        onehot_actions[range(batch_size), action.argmax(1).view(-1), ...] = 1

        # if action_mask is not None:
        #     onehot_actions[action_mask, ...] = 1

        inputs = torch.cat([board, onehot_actions, memory], 1)

        x = self.conv(inputs)
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        base = self.image_conv(x)
        memory = self.mem_gen(base)

        image = self.image_gen(base) + board

        reward = self.reward_conv(x)
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward)
        # reward = None

        return image, memory, reward

    def load_parameters(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

    def save_parameters(self, filepath):
        torch.save(self.state_dict(), filepath)

