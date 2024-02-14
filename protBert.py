import csv
from glob import glob
import os
import sys
from pathlib import Path
from transformers import TFBertModel, BertTokenizer,BertConfig
import re
import numpy as np
import tensorflow as tf

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
header = ['pdb','classification']
header.extend(range(1, max_seq_length))

file_features_seq = "feature_sequence_protBert.csv"
check_path = output_path+file_features_seq
print(check_path)
if not os.path.isfile(check_path):
    print("Creo il file")
    #Se il file non esiste, crea il file e scrivi l'header delle colonne
    with open(check_path, mode='w', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        csv_writer.writerow(header)

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = TFBertModel.from_pretrained("Rostlab/prot_bert", from_pt=True)


for aas_file in aas_files:
    with open(aas_file, mode='r', newline='') as aas:
        sequenceAAS = aas.readline().strip()

        sequence_input = ""
        for char in sequenceAAS:    
            sequence_input = sequence_input + " " + char 

        sequences_Example = [sequence_input]
        sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

        ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True, return_tensors="tf")
        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']

        embedding = model(input_ids)[0]
        embedding = np.asarray(embedding)
        attention_mask = np.asarray(attention_mask)

        features = [] 
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
        #np.set_printoptions(threshold=np.inf)  # Imposta il limite della visualizzazione delle righe a infinito
    
    #features = [array( [][] , dtype )]
    # Apri il file CSV in modalità append
    with open(check_path, mode='a', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        riga_da_scrivere = []
        pdb_id = Path(aas_file).stem  # Estrai il nome del file senza estensione
        riga_da_scrivere.append(pdb_id) # Inserisco nella prima colonna il pdb id
        riga_da_scrivere.append(get_pdb_class(pdb_id, sys.argv[3]))
        for row_np_arr in features[0]:
            average_feature = np.mean(row_np_arr).astype('float64')
            riga_da_scrivere.append(average_feature) # Calcolo della media delle feature di un amminoacido
        csv_writer.writerow(riga_da_scrivere)