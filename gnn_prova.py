# Install required packages.
import random
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

def gnn(dataset, num_classes):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    torch.manual_seed(12345)
    random.shuffle(dataset)
    train_dataset = dataset[:4]
    test_dataset = dataset[4:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    class GCN(torch.nn.Module):
        def __init__(self, num_features, hidden_channels, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            print("Shape of x before conv1:", x.shape)
            x = self.conv1(x, edge_index)
            print("Shape of x after conv1:", x.shape)
            x = x.relu()
            
            print("Shape of x before conv2:", x.shape)
            x = self.conv2(x, edge_index)
            print("Shape of x after conv2:", x.shape)
            x = x.relu()
            
            print("Shape of x before conv3:", x.shape)
            x = self.conv3(x, edge_index)
            print("Shape of x after conv3:", x.shape)
            x = global_mean_pool(x, batch)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
            return x


    # Calcola il numero di features per il primo grafo nel dataset
    first_data = dataset[0]
    num_features = first_data.num_node_features

    model = GCN(num_features=num_features, hidden_channels=5, num_classes=num_classes)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            print("Node features:")
            print(data.x[:10])  # Stampa le features dei primi 10 nodi
            print("Node labels:")
            print(data.y[:10])  # Stampa le etichette one-hot dei primi 10 nodi
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            print("Node features:")
            print(data.x[:10])  # Stampa le features dei primi 10 nodi
            print("Node labels:")
            print(data.y[:10])  # Stampa le etichette one-hot dei primi 10 nodi
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.



    for epoch in range(1, 171):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Esempio di utilizzo:
# Supponendo che dataset e num_classes siano già stati definiti in precedenza
# gnn(dataset, num_classes)
