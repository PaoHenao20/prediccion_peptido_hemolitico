from matplotlib import pyplot as plt
import pandas as pd
import glob
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess


# PyPept imports
from pyPept.sequence import Sequence, correct_pdb_atoms
from pyPept.molecule import Molecule
from collections import Counter
from typing import Literal, Dict
from rdkit import Chem
from mordred import Calculator, descriptors
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
import seaborn as sns
import json
from scipy.stats import shapiro, levene, ttest_ind, mode


import numpy as np

# Compatibilidad con versiones nuevas de NumPy
if not hasattr(np, 'bool'):
    np.bool = np.bool_   # alias a la versión nueva
if not hasattr(np, 'object'):
    np.object = np.object_   # alias correcto
if not hasattr(np, 'long'):
    np.long = np.int_   # 'long' era un alias de int en NumPy
if not hasattr(np, 'float'):
    np.float = float


warnings.filterwarnings("ignore", category=DeprecationWarning)



def generate_download_tasks(versions, splits, labels, raw_folder, base_url, final_path, output_name, extension):
    """
    Generates tuples of (url, local_filename, output_path) for each dataset.
    """
    tasks = []

    for version in versions: 
        if output_name == "HemoPI2":
             filename = f"{version}.{extension}"
             url = f"{base_url}/{final_path}/{filename }"
             output_dir = os.path.join(raw_folder, output_name)
             output_path = os.path.join(output_dir, filename)
             tasks.append((url, output_path))

        else:
            for split in splits:
                for label_name, label_code in labels.items():
                    filename = f"{version}_{split}_{label_name}.{extension}"
                    url = f"{base_url}/{final_path}/{version}/{split}/{label_code}.{extension}"
                    output_dir = os.path.join(raw_folder, output_name)
                    output_path = os.path.join(output_dir, filename)
                    tasks.append((url, output_path))


    return tasks


def clean_text_and_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by trimming whitespace from string columns and removing duplicate rows.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Strip leading/trailing spaces from string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df

def load_and_concatenate_csvs(folder: str, filenames: list) -> pd.DataFrame:
    dataframes = []
    for name in filenames:
        file_path = folder/f"{name}.csv"
        df = pd.read_csv(file_path, sep=",")
        print(f"{name}.csv loaded with shape: {df.shape}")
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def df_to_fasta(df, seq_col='sequence', output='peptides.fasta'):
    records = [
        SeqRecord(Seq(seq), id=f"peptide_{i+1}", description="")
        for i, seq in enumerate(df[seq_col])
    ]
    SeqIO.write(records, output, "fasta")

def run_cd_hit(input_fasta, output_prefix, identity):
    output_file = f"{output_prefix}_{int(identity * 100)}.fasta"
    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_file,
        "-c", str(identity)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"CD-HIT completado para identidad {identity*100:.0f}%. Resultado: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando CD-HIT con identidad {identity}: {e}")


def fasta_to_df(fasta_path: str, output_csv: str = None) -> pd.DataFrame:
    """Carga un archivo FASTA como DataFrame con una columna SEQUENCE"""
    from Bio import SeqIO
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
    
    df_result = pd.DataFrame({"SEQUENCE": sequences})

    if output_csv:
        df_result.to_csv(output_csv, index=False)
    return df_result

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def calc_kmer_frequencies(seq: str, k: int, relative: bool = True) -> Dict[str, float]:
    """Calculate sequence kmer frequencies"""
    seq = seq.upper().replace(" ", "")
    if not set(seq).issubset(VALID_AA):
        raise ValueError(f"Invalid amino acids in: {seq}")
    
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    
    if relative:
        total = sum(counts.values())
        return {kmer: count / total for kmer, count in counts.items()}
    return dict(counts)


def split_with_hyphens(cadena: str) -> str:
    if pd.notnull(cadena):
        return '-'.join(cadena)
    return cadena

def get_smiles(peptido: str) -> str:
    # 1) formatear la cadena
    seq_str = split_with_hyphens(peptido)
    print(seq_str)

    # 2) crear objeto Sequence y corregir átomos
    seq_obj = Sequence(seq_str)
    seq_obj = correct_pdb_atoms(seq_obj)

    # 3) generar molécula y convertir a ROMol
    mol_obj = Molecule(seq_obj)
    romol = mol_obj.get_molecule(fmt='ROMol')

    # 4) obtener SMILES
    smiles = Chem.MolToSmiles(romol)

    return smiles


def annotate_and_save(
    df: pd.DataFrame,
    seq_col: str = 'SEQUENCE',
    smiles_col: str = 'SMILES',
    output_csv: str = None
) -> pd.DataFrame:
    # 1) Guarda la lista de columnas originales
    original_cols = list(df.columns)
    # 2) Haz una copia para no mutar el DataFrame pasado
    df_out = df.copy()
    # 3) Añade la columna de SMILES
    df_out[smiles_col] = df_out[seq_col].apply(get_smiles)
    # 4) Reordena para que queden primero las originales y luego SMILES
    df_out = df_out[original_cols + [smiles_col]]
    # 5) Guarda si pidieron ruta
    if output_csv:
        df_out.to_csv(output_csv, index=False)
    return df_out

def compute_mordred_descriptors(df: pd.DataFrame,
                                smiles_col: str = 'SMILES') -> pd.DataFrame:
    """
    Para cada SMILES en df[smiles_col], calcula todos los descriptores de Mordred
    (ignore_3D=True) y devuelve un nuevo DataFrame con los descriptores.
    """
    # 1) Instanciar el calculador de Mordred
    calc = Calculator(descriptors, ignore_3D=True)

    # 2) Convertir cada SMILES a Mol
    mols = df[smiles_col].map(lambda s: Chem.MolFromSmiles(s))

    # 3) Ejecutar Mordred en lote y obtener DataFrame de descriptores
    #    calc.pandas acepta una lista/serie de RDKit Mol
    df_desc = calc.pandas(mols)

    # 4) Opcional: limpiar columnas con todos NaN o infinito
    df_desc = df_desc.dropna(axis=1, how='all')

    # 5) Resetear índices para alineación
    df_desc.index = df.index

    return df_desc