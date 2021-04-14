import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.mlp = nn.Sequential(nn.Linear(3 * 3 * 64, 120),
                                 nn.ReLU(),
                                 nn.Linear(120, 84),
                                 nn.ReLU(),
                                 nn.Linear(84, 10),
                                 nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.conv(x)
        bsz, nch, height, width = x.shape
        x = x.reshape(bsz, -1)

        y = self.mlp(x)
        return y
