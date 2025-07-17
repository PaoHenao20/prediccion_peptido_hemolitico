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

# RDKit imports
from rdkit import Chem
import pandas as pd
from mordred import Calculator, descriptors
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
import seaborn as sns
import json
from scipy.stats import shapiro, levene, ttest_ind, mode



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
    # df_sin_dup = df_sin_dup.head(50)

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
    df_clean = df_clean.head(5)

    # 5) Guardar a CSV si se especifica
    if output_csv:
        df_clean.to_csv(output_csv, index=False)

    return df_clean

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


def normalize_counts(X):
    """Normaliza las filas de una matriz para obtener frecuencias relativas."""
    row_sums = X.sum(axis=1)
    X_normalized = X.astype(float)
    for i in range(X.shape[0]):
        if row_sums[i, 0] != 0:
            X_normalized[i] /= row_sums[i, 0]
    return X_normalized

def compute_sequence_features(
    df: pd.DataFrame,
    seq_column: str = 'sequence',
    ks: List[int] = [1, 2, 3]
) -> pd.DataFrame:
    """
    Calcula frecuencias de aminoácidos, dipeptidos y tripeptidos para cada secuencia.
    Retorna un DataFrame con una fila por secuencia y columnas por k-mer.
    """
    result_df = df[[seq_column]].copy()

    for k in ks:
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
        sequences = df[seq_column].astype(str).str.upper().fillna("")
        X_counts = vectorizer.fit_transform(sequences)
        X_freqs = normalize_counts(X_counts)
        
        # Generar nombres de columnas
        kmer_cols = [f'{k}mer_{kmer}' for kmer in vectorizer.get_feature_names_out()]
        kmer_df = pd.DataFrame(X_freqs.toarray(), columns=kmer_cols)
        result_df = pd.concat([result_df.reset_index(drop=True), kmer_df], axis=1)

    return result_df

def plot_top_correlations(df_filtered, method, target_variables, output_dir):
    for target in target_variables:
            # Selecciona solo columnas numéricas
            numeric_df = df_filtered.select_dtypes(include=[np.number])
            
 
            corr = numeric_df.corr(method=method)[target].drop(target).sort_values(key=abs, ascending=False)
     
            # top 10
            top10 = corr.head(10)

            # Crear gráfico
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top10.values, y=top10.index, palette='viridis')
            plt.title(f'Top 10 variables más correlacionadas con {target} ({method.title()})')
            plt.xlabel(f'Correlación {method.title()}')
            plt.tight_layout()

            # Guardar figura
            filename = f"top10_{method}_{target}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            print(f"✅ Saved {filepath}")


def compute_statistical_summary_by_class(df, class_col='label', exclude_cols=None, output_dir=None, sample_size=5000):
    if exclude_cols is None:
        exclude_cols = []

    # Asegurarse de no incluir la variable de clase ni columnas excluidas
    numeric_cols = df.select_dtypes(include=np.number).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols + [class_col]]

    # Separar por clase
    classes = df[class_col].dropna().unique()
    if len(classes) != 2:
        raise ValueError("La función solo está preparada para clasificación binaria (2 clases).")

    class_values = {c: df[df[class_col] == c] for c in classes}
    results = []

    for col in feature_cols:
        col_stats = {"Variable": col}

        # Obtener valores por clase
        values = {c: class_values[c][col].dropna() for c in classes}

        for c in classes:
            sample = values[c].sample(min(len(values[c]), sample_size), random_state=42)
            col_stats[f"Mean_{c}"] = values[c].mean()
            col_stats[f"Median_{c}"] = values[c].median()
            col_stats[f"Q1_{c}"] = values[c].quantile(0.25)
            col_stats[f"Q3_{c}"] = values[c].quantile(0.75)
            col_stats[f"Shapiro_p_{c}"] = shapiro(sample)[1]

            moda = values[c].mode()
            col_stats[f"Mode_{c}"] = moda.iloc[0] if not moda.empty else np.nan

        # Levene test para varianzas iguales
        col_stats["Levene_p"] = levene(values[classes[0]], values[classes[1]])[1]

        results.append(col_stats)

    summary_df = pd.DataFrame(results)
    if output_dir:
        filename = "statistical.csv"
        filepath = os.path.join(output_dir, filename)
        summary_df.to_csv(filepath, index=False)

    return summary_df


def t_test_between_classes(df, class_col='label', equal_var=False, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    # Filtrar columnas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols + [class_col]]

    # Asegurarse de que hay solo 2 clases
    classes = df[class_col].dropna().unique()
    if len(classes) != 2:
        raise ValueError("Solo se admite clasificación binaria.")

    # Separar por clase
    df1 = df[df[class_col] == classes[0]]
    df2 = df[df[class_col] == classes[1]]

    results = []

    for col in feature_cols:
        x1 = df1[col].dropna()
        x2 = df2[col].dropna()

        if len(x1) < 2 or len(x2) < 2:
            continue  # saltar si no hay suficientes datos

        t_stat, p_val = ttest_ind(x1, x2, equal_var=equal_var)

        results.append({
            'Variable': col,
            f'Mean_{classes[0]}': x1.mean(),
            f'Mean_{classes[1]}': x2.mean(),
            'Mean_Diff': x1.mean() - x2.mean(),
            'p_value': p_val
        })

    return pd.DataFrame(results).sort_values('p_value')

def remove_highly_correlated_features(df, method='pearson', threshold=0.75, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    numeric_df = df.select_dtypes(include=np.number).drop(columns=exclude_cols, errors='ignore')
    corr_matrix = numeric_df.corr(method=method).abs()

    # Seleccionar solo la mitad superior para evitar duplicados
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identificar columnas con correlación mayor al umbral
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    df_reduced = df.drop(columns=to_drop, errors='ignore')

    print(f"🔍 Se eliminaron {len(to_drop)} variables por correlación > {threshold} usando {method}.")
    return df_reduced, to_drop