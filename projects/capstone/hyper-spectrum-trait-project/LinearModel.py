import torch.nn as nn
import torch.nn.functional as F



class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(2151,200)
        self.fc2 = nn.Linear(200,1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

