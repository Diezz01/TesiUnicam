#questo script crea il grafo da una matrice delle distanze e calcola le informazioni quantitative
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Carica la matrice delle distanze da un file CSV
df = pd.read_csv('1lwuC_aa_distance_matrix.csv', index_col=0)

# Crea un grafo non diretto
G = nx.Graph()

# Aggiungi nodi al grafo
G.add_nodes_from(df.index)

# Itera sulla matrice e aggiungi archi al grafo solo se sono sotto la diagonale principale
for i in range(len(df.index)):
    for j in range(i + 1, len(df.columns)):
        if not pd.isna(df.iloc[i, j]):
            G.add_edge(df.index[i], df.columns[j], weight=df.iloc[i, j])

# Disegna il grafo
pos = nx.spring_layout(G)  # Puoi scegliere un layout diverso a seconda delle tue esigenze
#nx.draw(G, pos, with_labels=True, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
#nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#print(G.number_of_edges())

num_nodi = G.number_of_nodes()
num_archi = G.number_of_edges()
grado_dei_nodi = dict(G.degree())
grado_medio = nx.average_degree_connectivity(G)
diametro_del_grafo = nx.diameter(G)
centralità_del_nodo = nx.degree_centrality(G)
coefficienti_di_clustering = nx.average_clustering(G)
componenti_connesse =nx.number_connected_components(G)
coeff_assortativita = nx.degree_assortativity_coefficient(G)


# Intestazione delle colonne
intestazione = ['Grafo_PDB', 'num_nodi', 'num_archi', 'grado_dei_nodi','grado_medio','diametro_del_grafo','centralità_del_nodo','coefficienti_di_clustering','componenti_connesse','coeff_assortativita']
# Creare un DataFrame con solo l'intestazione
df = pd.DataFrame(columns=intestazione)


# Lista di dizionari rappresentanti le righe
dati = [
    {'Grafo_PDB':"AA", 'num_nodi':num_nodi, 'num_archi':num_archi, 'grado_dei_nodi':grado_dei_nodi,'grado_medio':grado_medio,'diametro_del_grafo':diametro_del_grafo,'centralità_del_nodo':centralità_del_nodo,'coefficienti_di_clustering':coefficienti_di_clustering,'componenti_connesse':componenti_connesse,'coeff_assortativita':coeff_assortativita}
]

# Creare un DataFrame
df = pd.DataFrame(dati, columns=intestazione)

# Salva il DataFrame con i dati in un file CSV
df.to_csv('output.csv', index=False)
