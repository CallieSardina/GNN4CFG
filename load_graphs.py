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
from graphviz import Source

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

def load_dataset():
    # Read the list of binary files and generate CFGs
    with open("../bin_binaries_list.txt", "r") as file:
        binary_paths = file.read().splitlines()

    graph_list = []
    for binary_path in binary_paths:
        # Generate the edge indices and node features
        project = angr.Project('../' + binary_path[2:], auto_load_libs=False, main_opts={'backend': 'blob', 'arch': 'amd64'})
        # Get the CFG
        cfg = project.analyses.CFG()


        # save graph object to file
        #s = Source(cfg, filename=f'./cfg_data/graphviz{binary_path[5:]}.png', format="png")
        pickle.dump(cfg.graph, open(f'./cfg_data{binary_path[5:]}.pickle', 'wb'))

load_dataset()
#     labels = [0 for _ in range(len(graph_list))]        

#     return graph_list, labels

# def load_data_pt():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     with open("../bin_binaries_list_short.txt", "r") as file:
#         binary_paths = file.read().splitlines()

#     for binary_path in binary_paths:
#         pt_data = torch.load(f'./data/{binary_path[5:]}.pt', map_location=device)

# # loaded_data = load_data_pt()
# # loader = DataLoader(loaded_data, batch_size=32, shuffle=True)
# """
# Used the first time to generate the .pt files
# """
# graph_list, labels = load_dataset()
# dataset = data.Dataset(graph_list, labels)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Instantiate your GNN model
# model = GNNModel(in_channels=2, hidden_channels=16, out_channels=2)

# # Define loss function and optimizer
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Training loop
# model.train()
# for data in loader:
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = criterion(out, data.y)
#     loss.backward()
#     optimizer.step()

# # Evaluation loop
# model.eval()
# correct = 0
# for data in loader:
#     pred = model(data.x, data.edge_index).argmax(dim=1)
#     correct += int((pred == data.y).sum())

# # Calculate accuracy
# accuracy = correct / len(dataset)
# print(f'Accuracy: {accuracy:.4f}')
