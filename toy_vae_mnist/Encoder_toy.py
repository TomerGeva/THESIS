import torch.nn as nn
import torch.nn.functional as F


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
