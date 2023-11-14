import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from math import floor
import gcn_temp 
from torch_geometric.utils import add_self_loops

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_file_name_in_text_file(file_name, text_file_path):
    with open(text_file_path, 'r') as file:
        content = file.read()
        return file_name in content

def load_data(device):
    data = []
    for filename in os.listdir('./data'):
        if is_file_name_in_text_file(filename[:-3], 'small_binaries_500.txt'):
            data_file = os.path.join('./data', filename)
            if os.path.exists(data_file):
                print(data_file)
                data.append(torch.load(data_file))
            else:
                print("Something went wrong!")

    return data

# Load prediction based on model
data = load_data(device=device)

print(data[0].x.shape[0])

# Find the size of the largest graph
max_nodes = max(d.x.shape[0] for d in data) if data else 0
max_features = max(d.x.shape[1] for d in data) if data and data[0].x.dim() > 1 else 0
max_edges = max(d.edge_index.shape[1] for d in data) if data and len(data[0].edge_index) > 1 else 0

print()
print(max_nodes)
print(max_features)
print(max_edges)

for d in data:
    num_nodes_to_pad = max_nodes - d.x.shape[0]
    num_features_to_pad = max_features - d.x.shape[1]
    num_edges_to_pad = max_edges - d.edge_index.shape[1]

    # Padding the node features with zeros
    if num_nodes_to_pad > 0 or num_features_to_pad > 0:
        if num_nodes_to_pad > 0:
            padding_tensor = torch.zeros(num_nodes_to_pad, d.x.shape[1]).to(device)
            d.x = torch.cat([d.x, padding_tensor], dim=0)
        if num_features_to_pad > 0:
            padding_tensor = torch.zeros(d.x.shape[0], num_features_to_pad).to(device)
            d.x = torch.cat([d.x, padding_tensor], dim=1)

    # Padding the edge_index with self-loops and removing duplicates
    if num_edges_to_pad > 0:
        loop_edges = torch.tensor([[i for i in range(d.x.shape[0])],
                                   [i for i in range(d.x.shape[0])]], dtype=torch.long).to(device)
        d.edge_index = torch.cat([d.edge_index, loop_edges], dim=1)
        d.edge_index, _ = add_self_loops(d.edge_index)
        _, unique_idx = torch.unique(d.edge_index, dim=1)
        d.edge_index = d.edge_index[:, unique_idx]

# Split the data
valid_ratio = 0.1
test_ratio = 0.1
valid_size = 0#floor(len(data) * valid_ratio)
test_size = 2#floor(len(data) * test_ratio)
train_size = len(data) - valid_size - test_size
train_set, valid_set, test_set = torch.utils.data.random_split(data, lengths=[train_size, valid_size, test_size])

# Create DataLoader
batch_size = 1
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

model = gcn_temp.GNNModel(in_channels=max_features, hidden_channels=16, out_channels=2)
model.to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        x, edge_index, y = data.x.float().to(device), data.edge_index.to(device), data.y.to(device)

        # Forward pass
        output = model(x, edge_index)

        # Compute loss
        loss = criterion(output, y.view(-1))

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

# Evaluation Loop
print(len(train_set))
model.eval()
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation during evaluation
    for data in test_loader:  # Assuming you have a DataLoader for your test set
        x, edge_index, y = data.x.float().to(device), data.edge_index.to(device), data.y.to(device)
        print("HERE", x, edge_index)
        # Forward pass
        output = model(x, edge_index)

        # Get predictions
        _, predicted = torch.max(output, 1)

        # Update counts
        total += y.size(0)
        correct += (predicted == y).sum().item()

# Avoid division by zero error
accuracy = correct / total 
print(f'Accuracy: {accuracy:.4f}')