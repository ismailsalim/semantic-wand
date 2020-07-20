import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def build_model():
    model = TrimapGenerator()
    model.cuda()
    return model


class TrimapGenerator(nn.Module):
    def __init__(self):
        super(TrimapGenerator, self).__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class DataWrapper(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        if self.y is None:
             return self.X[index] 
        else:
             return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

