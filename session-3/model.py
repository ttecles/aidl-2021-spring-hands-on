import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(7*7*256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x
