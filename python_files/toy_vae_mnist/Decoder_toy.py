import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderToy(nn.Module):
    def __init__(self):
        super(DecoderToy, self).__init__()

        self.fc1 = nn.Linear(50, 300)
        self.fc2 = nn.Linear(300, 64 * 7 * 7)

        self.conv1  = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm1 = nn.BatchNorm2d(num_features=32)  # 7 --> 14
        self.conv2  = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)  # 14 X 14
        self.bnorm2 = nn.BatchNorm2d(num_features=16)
        self.conv3  = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)  # 28 X 28

    def forward(self, x):
        # ------------------------------
        # Fully Connected section
        # ------------------------------
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)  # reshaping

        # ------------------------------
        # Convolutional section
        # ------------------------------
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x
