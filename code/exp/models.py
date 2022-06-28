import sys

import numpy as np

sys.path.append("..")
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.tabnet import TabNet
from exp.utils import *


class FCNN(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.b2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.b3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 32)
        self.b4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(self.bn0(x)))
        x = self.b1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.b2(x)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.b3(x)
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.b4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


class FCNN_small(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, 4)
        self.b1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 4)
        self.b2 = nn.BatchNorm1d(4)
        self.fc3 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(self.bn0(x)))
        x = self.b1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.b2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FCNN_large(nn.Module):
    def __init__(self, inp_dim, width=256):
        super().__init__()
        self.b0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, width * 32)
        self.b1 = nn.BatchNorm1d(width * 32)
        self.fc2 = nn.Linear(width * 32, width * 4)
        self.b2 = nn.BatchNorm1d(width * 4)
        self.fc3 = nn.Linear(width * 4, width * 4)
        self.b3 = nn.BatchNorm1d(width * 4)
        self.fc5 = nn.Linear(width * 4, width * 4)
        self.b5 = nn.BatchNorm1d(width * 4)
        self.fc6 = nn.Linear(width * 4, width)
        self.b6 = nn.BatchNorm1d(width)
        self.fc7 = nn.Linear(width, 1)
        self.dropout = nn.Dropout(p=0.8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.b0(x)
        x = self.relu(self.fc1(x))
        x = self.b1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.b2(x)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.b3(x)
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.b5(x)
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        x = self.b6(x)
        x = self.dropout(x)
        x = self.fc7(x)
        return x


class FCNN2(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.b4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.b1(x)
        x = self.relu(self.fc2(x))
        x = self.b4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


class Perc(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.b0 = nn.BatchNorm1d(inp_dim)
        self.fc1 = nn.Linear(inp_dim, 1)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.b0(x)
        x = self.fc1(x)  # self.relu(self.fc1(x))
        return x


class TabNet_ieeecis(TabNet):
    def __init__(self, inp_dim, n_d, n_a, n_shared, n_ind, n_steps, relax, vbs):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=n_d,
            n_a=n_a,
            n_shared=n_shared,
            n_ind=n_ind,
            n_steps=n_steps,
            relax=relax,
            vbs=vbs,
        )


class TabNet_homecredit(TabNet):
    def __init__(self, inp_dim):
        super().__init__(
            inp_dim=inp_dim,
            final_out_dim=1,
            n_d=8,
            n_a=8,
            n_shared=3,
            n_ind=1,
            n_steps=3,
            relax=1.2,
            vbs=512,
        )
