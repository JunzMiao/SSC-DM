# -*- coding: UTF-8 -*-

from torch import nn

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        fc1_output = self.fc1(conv2_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        
        return fc3_output
