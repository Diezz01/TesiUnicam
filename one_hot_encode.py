# example of a one hot encoding
from collections import Counter
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import List
import gnn_prova
import nostra_gcn

def encode_aminoacids(node_char):
    # define data
    data = asarray([['A'], ['C'], ['D'], ['E'], ['F'], ['G'], ['H'], ['I'], ['K'], ['L'], ['M'], ['N'], ['P'], ['Q'], ['R'], ['S'],['T'],['U'],['V'],['W'],['Y']])
    # define one hot encoding
    encoder = OneHotEncoder()
    # transform data
    encoder.fit_transform(data)
   
    # Trasforma il carattere specifico in una matrice numpy di forma (1, 1)
    carattere_specifico_encoded = asarray([[node_char]])

    # Esegue l'encoding del carattere specifico
    encoding_carattere_specifico = encoder.transform(carattere_specifico_encoded)
    return encoding_carattere_specifico

def dataset_create(graphs_list: List[nx.Graph],graphs_classifications):
    dataset = []
    #creazione lista archi andata e ritorno indicizzati (da 0 a num_nodes-1) invece di etichette
    for single_graph in graphs_list:
        ar_edge_list = []
        node_list = list(single_graph.nodes())
        edge_list = single_graph.edges()
        for single_edge in edge_list:
            nodo1,nodo2 = single_edge
            tuple_indici_a = [node_list.index(nodo1),  node_list.index(nodo2)]
           
            ar_edge_list.append(tuple_indici_a)
            tuple_indici_b = [node_list.index(nodo2),  node_list.index(nodo1)]
            ar_edge_list.append(tuple_indici_b)
        #creo il tensore con le tuple di archi
        edge_index = torch.tensor(ar_edge_list, dtype=torch.long).t().contiguous()

        #creo la lista di amminoacidi codificati
        encoded_aminoacids = []
        
        for single_node in node_list:
            node_encode = (encode_aminoacids(single_node[0]))
            encoded_aminoacids.append(node_encode)
        
        encoded_aminoacids_array = [encoded_aminoacids[i].toarray()[0] for i in range(len(encoded_aminoacids))]
        #creo il tensore con gli ammioacidi codificati
        x = torch.tensor(encoded_aminoacids_array, dtype=torch.float)
        
        index_classification = graphs_classifications.index(single_graph.graph['label'])
        y = torch.tensor(index_classification)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.validate(raise_on_error=True)
        dataset.append(data)

    gnn_prova.gnn(dataset,len(graphs_classifications))
