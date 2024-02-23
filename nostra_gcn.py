import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

def train_gcn(dataset, num_classes):
    # Dividi il dataset in set di addestramento e set di test
    train_dataset = dataset[:int(0.8 * len(dataset))]  # Ad esempio, 80% per il training
    test_dataset = dataset[int(0.8 * len(dataset)):]   # Ad esempio, 20% per il test


    # Crea i data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Definisci il modello GCN
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(dataset[0].num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, num_classes)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv3(x, edge_index)
            x = global_mean_pool(x, batch)
            x = self.lin(x)
            return F.log_softmax(x, dim=-1)

    # Inizializza il modello
    model = GCN(hidden_channels=64, num_classes=num_classes)

    # Definisci l'ottimizzatore
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Definisci la funzione di perdita
    criterion = torch.nn.NLLLoss()

    # Addestra il modello
    def train():
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    # Valuta il modello sui dati di test
    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)

    # Esegui il training e la valutazione per un numero di epoche fissato
    for epoch in range(1, 51):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Ritorna il modello addestrato
    return model
