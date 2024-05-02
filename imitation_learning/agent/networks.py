import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN4(nn.Module):

    def __init__(self, history_length=0, n_classes=5):
        super(CNN4, self).__init__()

        self.c1 = nn.Conv2d(history_length+1, 16, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.c2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.r2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(18432, 128)
        self.r3 = nn.ReLU()

        self.fl = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.r1(self.c1(x))
        x = self.bn1(x)
        x = self.p1(x)

        x = self.r2(self.c2(x))
        x = self.bn2(x)
        x = self.p2(x)

        x = self.flat(x)
        x = self.r3(self.fc3(x))
        x = self.fl(x)
        return x
    