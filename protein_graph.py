import pandas as pd
import networkx as nx
from pathlib import Path
import os
from glob import glob
import csv


def graph_weights(distance_matrix, representation_type, threshold):
    if (representation_type == "sequence") :
        weights_matrix = distance_matrix.applymap(lambda x: 0)
    elif (representation_type == "contact_map") : 
        weights_matrix = distance_matrix.applymap(lambda x: 1 if  x <= threshold else 0)
    else:
        weights_matrix = distance_matrix.applymap(lambda x: 1/(1+x) if  x <= threshold else 0)

   # weights_matrix = weights_matrix.where(sequence_matrix == 0, 1)
    return weights_matrix

def create_graph(matrix_file):
    # Carica la matrice delle distanze da un file CSV
    df = pd.read_csv(matrix_file, index_col=0)
    threshold = 6.0e-10
    df = graph_weights(df,"",threshold)
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


def get_pdb_class(pdb_id, label_file):
    with open(label_file, mode='r') as label:
        reader = csv.reader(label, delimiter=';')
        for riga in reader:
            if(riga[0] == pdb_id):
                return riga[2]

def get_average_eccentricity(graph):
    eccentricities = nx.eccentricity(graph)
    # Calcolare l'eccentricità media
    average_eccentricity = sum(eccentricities.values()) / len(eccentricities)
    return average_eccentricity

#main
def main(input_path, label_file):
    file_graph = 'graph_info.csv'
    intestazione = ['pdb', 'num_nodi', 'num_archi','grado_medio','diametro_del_grafo','coefficienti_di_clustering','componenti_connesse','average_eccentricity','raggio_del_grafo','classification']

    # Verifica se il file esiste 
    check_path = input_path+"\\"+"graph"
    os.mkdir(check_path)
    check_path = check_path+"\\"+file_graph
    if not os.path.isfile(check_path):
        print("Creo il file")
        #Se il file non esiste, crea il file e scrivi l'intestazione delle colonne
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
            graph = create_graph(matrix_file)
            pdb_classification = get_pdb_class(pdb_id, label_file)
            #scrivo nel csv
            riga_da_scrivere = [f'{pdb_id}', f'{graph.number_of_nodes()}', 
                            f'{graph.number_of_edges()}', f'{int(sum(nx.average_degree_connectivity(graph)))}',
                            f'{nx.diameter(graph)}',
                            f'{nx.average_clustering(graph)}',
                            f'{nx.number_connected_components(graph)}',f'{get_average_eccentricity(graph)}',
                            f'{nx.radius(graph)}',
                            f'{pdb_classification}']
            csv_writer.writerow(riga_da_scrivere)
