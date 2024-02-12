# Install bio_embeddings using the command: pip install bio-embeddings[all]
import csv
import glob
import os
import sys
from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
from click import Path
import numpy as np
import torch 

if len(sys.argv) < 2: 
    print("USAGE: <output_dir>")
    exit()

seq = 'MVTYDFGSDEMHD' # A protein sequence of length L
#seq = 'KIIEKEKTADIDIIIIEIEATATSDSDVIVIRARAQSQSEVEVVRVRNQNQTETEAVAVKNKNKNGTGTNANAIAIKSGSGSNLILIQSQSGSGSDLDLVQVQQGADADLVLVQVQQQQEQEAEAA'
embedder = SeqVecEmbedder()
embedding = embedder.embed(seq)
protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
np_arr = protein_embd.cpu().detach().numpy()
np.set_printoptions(threshold=np.inf)  # Imposta il limite della visualizzazione delle righe a infinito
np.set_printoptions(precision=4, suppress=True)

print("lunghezza np arr: ",len(np_arr))

print(np_arr)
intestazione = range(1, len(seq))

#input_path = "/Users/filipporeucci/Desktop/"
output_path = sys.argv[1]
file_features_seq = "feature_sequence.csv"
check_path = output_path+file_features_seq
print(check_path)
if not os.path.isfile(check_path):
    print("Creo il file")
    #Se il file non esiste, crea il file e scrivi l'intestazione delle colonne
    with open(check_path, mode='w', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        csv_writer.writerow(intestazione)

# Apri il file CSV in modalità append
with open(check_path, mode='a', newline='') as file_csv:
    csv_writer = csv.writer(file_csv)
    riga_da_scrivere = []
    for row_np_arr in np_arr:
        sum = 0
        for singleFeature in row_np_arr:
            #scrivo nel csv ciascuna feature della sequenza
            sum = sum + singleFeature
        riga_da_scrivere.append(sum/1024)
    csv_writer.writerow(riga_da_scrivere)