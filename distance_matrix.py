#Questo script dato un file pdb calcola la matrice delle distanze
import re
import sys
from glob import glob
from Bio.PDB.PDBParser import PDBParser
import pandas as pd
import math
from pathlib import Path
pdb_parser = PDBParser(QUIET=True, PERMISSIVE=True)

_hydrogen = re.compile("[123 ]*H.*")

def is_hydrogen(atm):
    return _hydrogen.match(atm.get_id())


def is_hetero(res):
    return res.get_full_id()[3][0] != ' '


def residue_name(res):
    return "%s_%d_%s" % (res.get_full_id()[2], res.get_full_id()[3][1], res.get_full_id()[3][2])

def filterAtoms(arrayAtom, s):
    if len(s) == 0:
        return arrayAtom
    else:
        filtered_atoms = [atom for atom in arrayAtom if str(atom) == "<Atom {}>".format(s)]
        return filtered_atoms


def compute_distance_matrix(pdb_file, pdb_id, dest_path,checkAtom):
    # print(pdb_id)
    structure = pdb_parser.get_structure(pdb_id, pdb_file)
    # we filter out HETATM entries since they are not standard residue atoms
    residue_list = [res for res in structure.get_residues() if not is_hetero(res)]
    print (residue_list)
    res_names = [residue_name(res) for res in residue_list]
    print (res_names)
    distance_matrix = pd.DataFrame(0, index=res_names, columns=res_names)

    n_residues = len(residue_list)
    # precompute the list of heavy atoms (non-hydrogens) for each AA
    residue_atoms = {}
    for i in range(n_residues) :
        res = residue_list[i]
        heavy_atoms = [atom for atom in res.get_atoms() if not is_hydrogen(atom)]
        residue_atoms[i] = heavy_atoms

    for i in range(n_residues - 1):
        res1 = residue_list[i]
        n1 = residue_name(res1)
      # heavy_atoms_1 = residue_atoms[i]
        heavy_atoms_1 = filterAtoms(residue_atoms[i],checkAtom)
        for j in range(i + 1, n_residues):
            res2 = residue_list[j]
            n2 = residue_name(res2)
            #heavy_atoms_2 = residue_atoms[j]
            heavy_atoms_2 = filterAtoms(residue_atoms[j],checkAtom)
            print(heavy_atoms_1)
            min_distance = math.inf
            for atm1 in heavy_atoms_1:
                for atm2 in heavy_atoms_2:
                  
                    distance = atm1 - atm2
                    if distance < min_distance:
                        min_distance = distance

                distance_matrix.loc[n1, n2] = min_distance
                distance_matrix.loc[n2, n1] = min_distance
    # print(distance_matrix)
    distance_matrix.to_csv(f"{dest_path}/{pdb_id}_aa_distance_matrix.csv.gz", compression='gzip')


if len(sys.argv) >= 5:
    compute_distance_matrix(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
else:
    compute_distance_matrix(sys.argv[1], sys.argv[2], sys.argv[3], "")