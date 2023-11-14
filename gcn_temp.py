import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import data
import angr
import os
import torch.nn.functional as F

import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        # Linear transformation for features
        self.linear_features = torch.nn.Linear(in_channels - 1, hidden_channels)

        # Graph convolutional layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Linear transformation for features
        x_features = self.linear_features(x[:, 1:])

        # Print shape for debugging
        print("x_features shape:", x_features.shape)

        # Apply graph convolutional layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Pooling over all nodes in the graph (e.g., global mean pooling)
        x = torch.mean(x, dim=0, keepdim=True)

        # Assuming this is a classification task with log_softmax
        return F.log_softmax(x, dim=1)

