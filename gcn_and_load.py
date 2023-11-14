import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import data
import angr
import os

def load_dataset():
    # Read the list of binary files and generate CFGs
    with open("../bin_binaries_list_short.txt", "r") as file:
        binary_paths = file.read().splitlines()

    graph_list = []
    for binary_path in binary_paths:
        print(binary_path)

        project = angr.Project('../' + binary_path[2:], auto_load_libs=False, main_opts={'backend': 'blob', 'arch': 'amd64'})
        cfg = project.analyses.CFG()
        graph_list.append(cfg.graph)

        nodes = list(cfg.graph.nodes())
        edges = list(cfg.graph.edges())
        node_to_index = {node: index for index, node in enumerate(nodes)}
        edge_index = [[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        node_features = torch.eye(len(nodes))  

        data = Data(x=node_features, edge_index=edge_index, y=0)
        torch.save(data, f'./data/{binary_path[5:]}.pt')


    labels = [0 for _ in range(len(graph_list))]        

    return graph_list, labels

def load_data_pt():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("../bin_binaries_list_short.txt", "r") as file:
        binary_paths = file.read().splitlines()

    for binary_path in binary_paths:
        pt_data = torch.load(f'./data/{binary_path[5:]}.pt', map_location=device)

