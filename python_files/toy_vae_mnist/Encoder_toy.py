import torch.nn as nn
import torch.nn.functional as F


class EncoderToy(nn.Module):
    def __init__(self):
        super(EncoderToy, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=5, stride=1, padding=2)  # 28 --> 14
        self.bnorm1 = nn.BatchNorm2d(num_features=16)
        self.conv2  = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)  # 14 --> 7
        self.bnorm2 = nn.BatchNorm2d(32)
        self.conv3  = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)  # 7  --> 7
        self.bnorm3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 300)
        self.fc2 = nn.Linear(300, 100)

    def forward(self, x):
        # ------------------------------
        # Convolutional section
        # ------------------------------
        x = self.pool(F.relu(self.bnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.bnorm2(self.conv2(x))))
        x = self.bnorm3(self.conv3(x))

        # ------------------------------
        # Fully Connected section
        # ------------------------------
        x = x.view(-1, 64 * 7 * 7)  # reshaping
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
