import os
from pathlib import Path
import subprocess
import pandas as pd


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

def load_and_concatenate_csvs(folder: Path, filenames: list) -> pd.DataFrame:
    dataframes = []
    for name in filenames:
        file_path = folder/f"{name}.csv"
        df = pd.read_csv(file_path, sep=",")
        print(f"{name}.csv loaded with shape: {df.shape}")
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


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