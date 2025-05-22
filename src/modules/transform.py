import pandas as pd
import glob
import os

# PyPept imports
from pyPept.sequence import Sequence, correct_pdb_atoms
from pyPept.molecule import Molecule

# RDKit imports
from rdkit import Chem
import pandas as pd
from mordred import Calculator, descriptors
import numpy as np
import warnings

# Parche para compatibilidad con versiones antiguas de librerías
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'long'):
    np.long = int  # Python 3 ya no tiene 'long'

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def clean_df(df: pd.DataFrame, output_csv: str = None) -> pd.DataFrame:
    # 1) Copiar para no mutar el original
    df_clean = df.copy()

    # 2) Eliminar duplicados
    df_clean = df_clean.drop_duplicates(ignore_index=True)

    # 3) Quitar espacios al inicio/final en 'SEQUENCE'
    df_clean['SEQUENCE'] = df_clean['SEQUENCE'].astype(str).str.strip()

    # 4) Eliminar filas donde 'SEQUENCE' sea NaN o esté vacío
    df_clean = df_clean[df_clean['SEQUENCE'].notna()]           # no NaN
    df_clean = df_clean[df_clean['SEQUENCE'] != '']             # no vacías
    df_clean = df_clean.reset_index(drop=True)

    # 5) Guardar a CSV si se especifica
    if output_csv:
        df_clean.to_csv(output_csv, index=False)

    return df_clean




def separar_con_guiones(cadena: str) -> str:
    if pd.notnull(cadena):
        return '-'.join(cadena)
    return cadena



def get_smiles(peptido: str) -> str:
    # 1) formatear la cadena
    seq_str = separar_con_guiones(peptido)
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