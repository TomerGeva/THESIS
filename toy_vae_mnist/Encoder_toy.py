import math
import torch.nn as nn
import torch.nn.functional as F
from neural_network_block_classes import DenseTransitionBlock, DenseBlock


class EncoderToy(nn.Module):
    def __init__(self):
        super(EncoderToy, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=1,  out_channels=128, kernel_size=5, stride=1, padding=2)   # 28 --> 14
        self.bnorm1 = nn.BatchNorm2d(num_features=128)
        self.conv2  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)  # 14 --> 7
        self.bnorm2 = nn.BatchNorm2d(256)
        self.conv3  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=1, padding=1)  # 7  --> 3
        self.bnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=0)  # 3  --> 1

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1024, 300)
        self.fc2 = nn.Linear(300, 100)

    def forward(self, x):
        # ------------------------------
        # Convolutional section
        # ------------------------------
        x = self.pool(F.leaky_relu(self.bnorm1(self.conv1(x)), 0.01))
        x = self.pool(F.leaky_relu(self.bnorm2(self.conv2(x)), 0.01))
        x = self.pool(F.leaky_relu(self.bnorm3(self.conv3(x)), 0.01))
        x = F.leaky_relu(self.conv4(x), 0.01)

        # ------------------------------
        # Fully Connected section
        # ------------------------------
        x = x.view(-1, 1024)  # reshaping
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DenseEncoderToy(nn.Module):
    def __init__(self):
        super(DenseEncoderToy, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=1,  out_channels=3, kernel_size=3, stride=1, padding=1)
        self.dense1 = DenseBlock(channels=3, depth=4, growth_rate=40, kernel_size=3, stride=1, padding=1)
        self.trans1 = DenseTransitionBlock(in_channels=3+4*40, out_channels=math.floor(163/2), kernel_size=3, stride=1,
                                           padding=1)
        self.dense2 = DenseBlock(channels=81, depth=4, growth_rate=40, kernel_size=3, stride=1, padding=1)
        self.trans2 = DenseTransitionBlock(in_channels=81+4*40, out_channels=math.floor(241/2), kernel_size=3, stride=1,
                                           padding=1)
        self.dense3 = DenseBlock(channels=120, depth=4, growth_rate=40, kernel_size=3, stride=1, padding=1)
        self.trans3 = DenseTransitionBlock(in_channels=120+4*40, out_channels=math.floor(280/2), kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.dense4 = DenseBlock(channels=140, depth=4, growth_rate=40, kernel_size=3, stride=1, padding=1)
        self.trans4 = DenseTransitionBlock(in_channels=140+4*40, out_channels=math.floor(300/2), kernel_size=3, stride=1,
                                           padding=1)

        self.zp = nn.ZeroPad2d(1)

        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 100)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.zp(out)
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.trans4(self.dense4(out))

        # ------------------------------
        # Fully Connected section
        # ------------------------------
        out = out.view(-1, 150 * 4)  # reshaping
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
