import pandas as pd
import networkx as nx
from pathlib import Path
import os
from glob import glob
import csv
import numpy as np
from scipy.sparse import csr_matrix, block_diag
from scipy.sparse import find
import one_hot_encode
from collections import Counter

#legenda che rappresenta il tipo di nodo nel dataset
def switch_iupac(code):
    switcher = {
        'A': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'K': 9,
        'L': 10,
        'M': 11,
        'N': 12,
        'Q': 13,
        'R': 14,
        'S': 15,
        'T': 16,
        'U': 17,
        'V': 18,
        'W': 19,
        'Y': 20
    }
    return switcher.get(code.upper(), -1)  # Ritorna -1 se il codice non è presente nel dizionario

def get_sparse_matrix(adjacent_matrixs, card_matrix):
    sparse_matrix = csr_matrix((card_matrix, card_matrix), dtype=np.float64)  # Creazione di una matrice sparsa vuota
    start = 0
    for ad_m in adjacent_matrixs:
        ad_m_rows, ad_m_cols = ad_m.shape  # Otteniamo le dimensioni della matrice sparsa
        sparse_matrix[start:start+ad_m_rows, start:start+ad_m_cols] = ad_m  # Assegnamento della matrice sparsa alla posizione corretta nella matrice composta
        start += ad_m_rows
    return sparse_matrix
       
def apply_threshold(x, i, j,threshold):
    if i == j:
        # Se l'elemento è sulla diagonale, restituisci il valore originale
        return x
    else:
        # Altrimenti, applica il threshold
        return 1 if x <= threshold else 0

def graph_weights(distance_matrix, threshold):
    thresholded_matrix = pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns)
    for i, row in enumerate(distance_matrix.index):
        for j, col in enumerate(distance_matrix.columns):
            thresholded_matrix.loc[row, col] = apply_threshold(distance_matrix.loc[row, col], i, j,threshold)
    return thresholded_matrix
    
def create_graph(matrix_file):
    # Carica la matrice delle distanze da un file CSV
    df = pd.read_csv(matrix_file, index_col=0)
    threshold = 4.0e-10
    df = graph_weights(df,threshold)
    print (df)
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
        reader = csv.reader(label, delimiter=',')
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
    dataset_path = input_path+"\\"+"DataSet"
    os.mkdir(dataset_path)
   
# Apri il file CSV in modalità append
    with open(check_path, mode='a', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        matrix_files = glob(os.path.join(input_path, '*.csv'))
        graphs_classifications = []
        nodes_labels = []
        nodes_for_graphs = []
        adjacent_matrixs = []
        count = 1

        graphs_list = []

        for matrix_file in matrix_files:
            pdb_id = Path(matrix_file).stem  #Estrai il nome del file senza estensione
            print("Analyzing: ",pdb_id)
            graph = create_graph(matrix_file)
           
            pdb_classification = get_pdb_class(pdb_id, label_file)
            graph.graph['label'] = pdb_classification
            #scrivo nel csv
            riga_da_scrivere = [f'{pdb_id}', f'{graph.number_of_nodes()}', 
                            f'{graph.number_of_edges()}', f'{int(sum(nx.average_degree_connectivity(graph)))}',
                            f'{nx.diameter(graph)}',
                            f'{nx.average_clustering(graph)}',
                            f'{nx.number_connected_components(graph)}',f'{get_average_eccentricity(graph)}',
                            f'{nx.radius(graph)}',
                            f'{pdb_classification}']
            csv_writer.writerow(riga_da_scrivere)
            graphs_list.append(graph)
            #rnella sezione che segue venegono raccolti tutti i dati per la creazione del dataset per le gnn
            graphs_classifications.append(pdb_classification)#raccolgo tutti i nodi di tutti i grafi
            for node in graph.nodes():
                nodes_labels.append(switch_iupac(node[0]))
                nodes_for_graphs.append(count)

            adjacent_matrixs.append(nx.adjacency_matrix(graph))
            count += 1
        one_hot_encode.dataset_create(graphs_list,len(Counter(graphs_classifications)))



    #registro le classificazioni dei grafi
    with open(dataset_path+"\\"+"DATASET_graph_labels.txt", "w") as file:
        for number in graphs_classifications:
            file.write(str(number) + "\n")

    #registro le eticchette dei nodi
    with open(dataset_path+"\\"+"DATASET_node_labels.txt", "w") as file:
        for number in nodes_labels:
            file.write(str(number) + "\n")

    with open(dataset_path+"\\"+"DATASET_graph_indicator.txt", "w") as file:
        for number in nodes_for_graphs:
            file.write(str(number) + "\n")
   
    sparse_matrix=get_sparse_matrix(adjacent_matrixs,len(nodes_for_graphs))
    posizioni_valori_non_zero = sparse_matrix.nonzero()

    # Convertiamo le posizioni in una lista di coppie di indici
    index_pairs = list(zip(*posizioni_valori_non_zero))

    with open(dataset_path+"\\"+"DATASET_A.txt", "w") as file:
            for pair in index_pairs:
                file.write(f"{pair[0]+1}, {pair[1]+1}\n")