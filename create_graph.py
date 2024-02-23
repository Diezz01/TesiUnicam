import networkx as nx
import numpy as np
import one_hot_encode

lista_grafi = []
G = nx.Graph()
G.add_node(1, label='A')
G.add_node(2, label='Y')
G.add_node(3, label='C')
G.add_node(4, label='D')
G.graph['label'] = '1'
#G.add_nodes_from(range(4))
G.add_edges_from([(1, 2), (2, 4), (4, 3), (1, 3)])
print("adm 1:\n",nx.adjacency_matrix(G).toarray())
lista_grafi.append(G)

G2 = nx.Graph()
G2.add_node(1, label='F')
G2.add_node(2, label='G')
G2.add_node(3, label='J')
G2.add_node(4, label='E')
G2.add_node(5, label='W')
G2.graph['label'] = '2'
G2.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4), (2, 4), (4, 5)])
print("adm 2:\n",nx.adjacency_matrix(G2).toarray())
lista_grafi.append(G2)

G3 = nx.Graph()
G3.add_node(1, label='M')
G3.add_node(2, label='H')
G3.add_node(3, label='C')
G3.add_node(4, label='E')
G3.add_node(5, label='D')
G3.graph['label'] = '3'
G3.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 5), (4, 5)])
print("adm 3: \n",nx.adjacency_matrix(G3).toarray())
lista_grafi.append(G3)
# Grafo con 4 nodi e 4 archi
G4 = nx.Graph()
G4.add_node(1, label='A')
G4.add_node(2, label='C')
G4.add_node(3, label='K')
G4.add_node(4, label='M')
G4.graph['label'] = '4'
G4.add_edges_from([(1, 2), (3, 4), (4, 1)])
print("adm 4:\n",nx.adjacency_matrix(G4).toarray())
lista_grafi.append(G4)
# Grafo con 4 nodi e 4 archi
G5 = nx.Graph()
G5.add_node(1, label='A')
G5.add_node(2, label='C')
G5.add_node(3, label='Y')
G5.add_node(4, label='M')
G5.add_node(5 ,label='G')
G5.graph['label'] = '5'
G5.add_edges_from([(1, 2), (1, 4), (1, 3), (2, 5), (3, 5), (4, 5)])
print("adm 5:\n",nx.adjacency_matrix(G5).toarray())
lista_grafi.append(G5)

one_hot_encode.dataset_create(lista_grafi,3)
