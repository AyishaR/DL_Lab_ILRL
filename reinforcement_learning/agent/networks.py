import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    """
    CartPole network
    """
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    """
    CarRacing network
    """
    def __init__(self, history_length=0, n_classes=5):
        super(CNN, self).__init__()

        self.c1 = nn.Conv2d(history_length+1, 32, kernel_size=(3,3), stride=1, padding=1)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.c2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.r2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(18432, 512)
        self.r3 = nn.ReLU()

        self.d3 = nn.Dropout(0.1)
        self.fl = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.r1(self.c1(x))
        x = self.p1(x)

        x = self.r2(self.c2(x))
        x = self.p2(x)

        x = self.flat(x)

        x = self.r3(self.fc3(x))
        x = self.d3(x)
        x = self.fl(x)
        return x
