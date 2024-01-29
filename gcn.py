from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        # x: input features for nodes
        # adjacency_matrix: adjacency matrix representing the graph structure
        
        # Perform graph convolution
        x = torch.matmul(adjacency_matrix, x)
        x = self.linear(x)
        x = F.relu(x)
        return x

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GraphConvolutionalNetwork, self).__init__()
        self.layer1 = GraphConvolutionLayer(num_features, hidden_size)
        self.layer2 = GraphConvolutionLayer(hidden_size, num_classes)

    def forward(self, x, adjacency_matrix):
        x = self.layer1(x, adjacency_matrix)
        x = self.layer2(x, adjacency_matrix)
        return F.log_softmax(x, dim=1)
    
if len(sys.argv) != 2: 
    print("Usage: python script.py <input_matrix_file>")
    exit()

# Define your input features and load adjacency matrix from a CSV file
num_nodes = 13
num_features = 6
num_classes = 5
hidden_size = 16

# Random input features (you should replace this with your data)
input_features = torch.randn(num_nodes, num_features)

# Load adjacency matrix from a CSV file
#adjacency_matrix_file = "/Users/filipporeucci/Desktop/output/1lwuC.csv"
adjacency_matrix_file = sys.argv[1]

adjacency_matrix_df = pd.read_csv(adjacency_matrix_file, delimiter=',')  # Assuming the CSV file has a header

# Convert string values to float
adjacency_matrix_df = adjacency_matrix_df.map(lambda x: float(x.split(';')[0]) if isinstance(x, str) else 0)

# Convert the DataFrame to a tensor
adjacency_matrix = torch.tensor(adjacency_matrix_df.values, dtype=torch.float)

# Normalize the adjacency matrix (optional)
adjacency_matrix = F.normalize(adjacency_matrix, p=1, dim=1)

# Create the GCN model
gcn_model = GraphConvolutionalNetwork(num_features, hidden_size, num_classes)

# Forward pass
output = gcn_model(input_features, adjacency_matrix)

print("GCN Output:")
print(output)
