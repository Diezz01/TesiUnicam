from Bio import PDB
from Bio.PDB.Polypeptide import PPBuilder

# Definisci il percorso del file PDB
file_pdb = "C:\\Users\\diego\\Desktop\\PROTEINE\\1a17A.pdb"

# Crea un parser per il file PDB
parser = PDB.PDBParser()

# Carica la struttura del file PDB
struttura = parser.get_structure("struttura", file_pdb)

# Inizializza un costruttore di polipeptidi
builder = PPBuilder()

# Inizializza una lista per la sequenza aminoacidica
sequenza = []

# Estrai la sequenza aminoacidica per ogni catena
for modello in struttura:
    for chain in modello:
        polipeptide = builder.build_peptides(chain)
        for pp in polipeptide:
            sequenza.append(pp.get_sequence())

# Unisci le sequenze aminoacidiche
sequenza_aminoacidica = "".join(str(seq) for seq in sequenza)

# Stampa la sequenza aminoacidica ottenuta
print("Sequenza aminoacidica ottenuta:", sequenza_aminoacidica)
