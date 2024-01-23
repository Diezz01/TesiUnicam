import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob
import sys
import csv

def create_graph(matrix_file, pdb_id):
    # Carica la matrice delle distanze da un file CSV
    df = pd.read_csv(matrix_file, index_col=0)

    # Crea un grafo non diretto
    G = nx.Graph()

    # Aggiungi nodi al grafo
    G.add_nodes_from(df.index)

    # Itera sulla matrice e aggiungi archi al grafo solo se sono sotto la diagonale principale
    for i in range(len(df.index)):
        for j in range(i + 1, len(df.columns)):
            if not pd.isna(df.iloc[i, j]):
                G.add_edge(df.index[i], df.columns[j], weight=df.iloc[i, j])
    return G

#main
input_path = sys.argv[1]
output_path = sys.argv[2]

file_graph = 'graph_info.csv'
intestazione = ['PDB', 'num_nodi', 'num_archi','grado_medio','diametro_del_grafo','coefficienti_di_clustering','componenti_connesse']

# Verifica se il file esiste
check_path = output_path+"\\"+file_graph
print(check_path)
if not os.path.isfile(check_path):
    print("Creo il file")
    # Se il file non esiste, crea il file e scrivi l'intestazione delle colonne
    with open(check_path, mode='w', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        csv_writer.writerow(intestazione)

# Apri il file CSV in modalità append
with open(check_path, mode='a', newline='') as file_csv:
    csv_writer = csv.writer(file_csv)
    matrix_files = glob(os.path.join(input_path, '*.csv'))
    for matrix_file in matrix_files:
        pdb_id = Path(matrix_file).stem  # Estrai il nome del file senza estensione
        print("Analyzing: ",pdb_id)
        graph = create_graph(matrix_file, pdb_id)
        #scrivo nel csv
        riga_da_scrivere = [f'{pdb_id}', f'{graph.number_of_nodes()}', 
                            f'{graph.number_of_edges()}', f'{nx.average_degree_connectivity(graph)}',
                            f'{nx.diameter(graph)}',
                            f'{nx.average_clustering(graph)}',
                            f'{nx.number_connected_components(graph)}']
        csv_writer.writerow(riga_da_scrivere)