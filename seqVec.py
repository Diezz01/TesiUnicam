# Install bio_embeddings using the command: pip install bio-embeddings[all]
import csv
from glob import glob
import os
import sys
from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
from click import Path
from pathlib import Path
import numpy as np
import torch 

def getMaxSeqLength(aas_files):
    max = 0
    for aas_file in aas_files:
        with open(aas_file, mode='r', newline='') as aas:
            seq = aas.readline()
            if len(seq) > max:
                max = len(seq)
    return max

def get_pdb_class(pdb_id, label_file):
    with open(label_file, mode='r') as label:
        reader = csv.reader(label, delimiter=',')
        for riga in reader:
            if(riga[0] == pdb_id):
                return riga[2]

if len(sys.argv) < 3: 
    print("USAGE: <input_dir> <output_dir> <label_file>")
    exit()

input_path = sys.argv[1]
output_path = sys.argv[2]
aas_files = glob(os.path.join(input_path, '*.txt')) #prendo tutti i file con estensione aas dalla directory fornita

max_seq_length = getMaxSeqLength(aas_files)
intestazione = range(0, max_seq_length+1)

file_features_seq = "feature_sequence.csv"
check_path = output_path+file_features_seq
print(check_path)
if not os.path.isfile(check_path):
    print("Creo il file")
    #Se il file non esiste, crea il file e scrivi l'intestazione delle colonne
    with open(check_path, mode='w', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        csv_writer.writerow(intestazione)

for aas_file in aas_files:
    with open(aas_file, mode='r', newline='') as aas:
        seq = aas.readline().strip()

    embedder = SeqVecEmbedder()
    embedding = embedder.embed(seq)
    protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
    np_arr = protein_embd.cpu().detach().numpy()
    #np.set_printoptions(threshold=np.inf)  # Imposta il limite della visualizzazione delle righe a infinito
    np.set_printoptions(precision=4, suppress=True)

    print(len(np_arr))

    # Apri il file CSV in modalità append
    with open(check_path, mode='a', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        riga_da_scrivere = []
        pdb_id = Path(aas_file).stem  # Estrai il nome del file senza estensione
        riga_da_scrivere.append(pdb_id) # Inserisco nella prima colonna il pdb id
        riga_da_scrivere.append(get_pdb_class(pdb_id, sys.argv[3]))
        for row_np_arr in np_arr:
            sum = 0
            for singleFeature in row_np_arr:
                #scrivo nel csv ciascuna feature della sequenza
                sum = sum + singleFeature
            riga_da_scrivere.append(sum/1024)
        csv_writer.writerow(riga_da_scrivere)