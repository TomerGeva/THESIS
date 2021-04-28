import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderToy(nn.Module):
    def __init__(self):
        super(DecoderToy, self).__init__()

        self.conv1  = nn.ConvTranspose2d(in_channels=50, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.conv2  = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm2d(num_features=256)   # 2 --> 4
        self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bnorm3 = nn.BatchNorm2d(num_features=128)   # 4 --> 7
        self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm2d(num_features=32)    # 7 --> 14
        self.conv5  = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)  # 28 X 28

    def forward(self, x):
        # ------------------------------
        # Convolutional section
        # ------------------------------
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.bnorm2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bnorm3(self.conv3(x)), 0.01)
        x = F.leaky_relu(self.bnorm4(self.conv4(x)), 0.01)
        x = torch.tanh(self.conv5(x))
        return x
