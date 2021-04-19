import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, conv_kern=64, mlp_neurons=120, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, int(conv_kern/2), 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(int(conv_kern/2), conv_kern, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(conv_kern, conv_kern, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(6*6*conv_kern, mlp_neurons),
            nn.ReLU(),
            nn.Linear(mlp_neurons, int(mlp_neurons/2)),
            nn.ReLU(),
            nn.Linear(int(mlp_neurons/2), 16),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        bsz, nch, height, width = x.shape
        x = x.reshape(bsz, -1)
        x = self.mlp(x)
        return x
