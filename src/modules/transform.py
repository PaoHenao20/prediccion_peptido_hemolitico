import pandas as pd
import glob
import os

# PyPept imports
from pyPept.sequence import Sequence, correct_pdb_atoms
from pyPept.molecule import Molecule

# RDKit imports
from rdkit import Chem


def concat_csv_sin_duplicados(folder_path: str,
                              pattern: str = '*.csv',
                              output_path: str = None) -> pd.DataFrame:

    # 1) Generar lista de rutas
    glob_path = os.path.join(folder_path, pattern)
    archivos = glob.glob(glob_path)
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos con el patrón {glob_path}")

    # 2) Leer cada CSV en un DataFrame
    dfs = []
    for ruta in archivos:
        df = pd.read_csv(ruta)
        dfs.append(df)

    # 3) Concatenar y eliminar duplicados
    df_concat = pd.concat(dfs, ignore_index=True)
    df_sin_dup = df_concat.drop_duplicates()

    # 4) Guardar
    if output_path:
        df_sin_dup.to_csv(output_path, index=False)

    return df_sin_dup



def separar_con_guiones(cadena: str) -> str:
    if pd.notnull(cadena):
        cleaned = cadena.replace('-', '')
        return '-'.join(cleaned)
    return cadena



def get_smiles(peptido: str) -> str:
    # 1) formatear la cadena
    seq_str = separar_con_guiones(peptido)

    # 2) crear objeto Sequence y corregir átomos
    seq_obj = Sequence(seq_str)
    seq_obj = correct_pdb_atoms(seq_obj)

    # 3) generar molécula y convertir a ROMol
    mol_obj = Molecule(seq_obj)
    romol = mol_obj.get_molecule(fmt='ROMol')

    # 4) obtener SMILES
    smiles = Chem.MolToSmiles(romol)

    return smiles
    
