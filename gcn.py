import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import data
import angr
import os
import pickle
from math import floor
from graphviz import Source
import networkx as nx

# Replace 'directory_path' with the path to your directory
directory_path = './cfg_data/pickles'

# List all files in the directory
file_list = os.listdir(directory_path)

cfg_list = []
# Iterate over each file in the list
for file_name in file_list:
    # Create the file path
    file_path = os.path.join(directory_path, file_name)

    # Load the file
    # cfg_list.append(pickle.load(open(f'{file_path}', 'rb')))
    G = pickle.load(open(f'{file_path}', 'rb'))
    # Get the adjacency matrix as a numpy array
    adjacency_matrix = nx.adjacency_matrix(G).todense()

    # Convert the adjacency matrix to a PyTorch tensor
    tensor_representation = torch.tensor(adjacency_matrix)
    cfg_list.append(tensor_representation)

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

def split_data(dataset, valid_ratio=0.1, test_ratio=0.1):
    valid_size = floor(len(dataset) * valid_ratio)
    test_size = floor(len(dataset) * test_ratio)
    train_size = len(dataset) - valid_size - test_size
    splits = torch.utils.data.random_split(dataset, lengths=[train_size, valid_size, test_size])

    return splits

# Convert the data to a tensor before passing it to the DataLoader
data = cfg_list 

loader = DataLoader(data, batch_size=8, shuffle=True)
train_set, valid_set, test_set = split_data(data)

# train_set = torch.tensor(train_set)  # Convert the sets to tensors before passing them to the DataLoader
# valid_set = torch.tensor(valid_set)
# test_set = torch.tensor(test_set)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=8)
test_loader = DataLoader(test_set, batch_size=8)

# Instantiate your GNN model
model = GNNModel(in_channels=2, hidden_channels=16, out_channels=2)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for data in train_loader:
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

# Evaluation loop
model.eval()
correct = 0
for data in eval_loader:
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct += int((pred == data.y).sum())

# Calculate accuracy
accuracy = correct / len(dataset)
print(f'Accuracy: {accuracy:.4f}')


# Define loss function and optimizer
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# train(model, train_loader, optimizer, criterion)  # Pass the model to the train function

# correct = eval(model, test_loader)  # Pass the model to the eval function

# # Calculate accuracy
# accuracy = correct / len(data)
# print(f'Accuracy: {accuracy:.4f}')