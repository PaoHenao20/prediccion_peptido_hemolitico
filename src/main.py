import os
from pathlib import Path
from venv import logger
from Bio import SeqIO 

import pandas as pd
from datetime import datetime
from data_processing.extractor import download_file_from_url
from data_processing.transformation import (
    generate_download_tasks,
    clean_text_and_remove_duplicates,
    load_and_concatenate_csvs,
    df_to_fasta,
    run_cd_hit,
    fasta_to_df,
    calc_kmer_frequencies,
    annotate_and_save,
    compute_mordred_descriptors)

""" 
url Reference" https://webs.iiitd.edu.in/raghava/hemopi/datasets.php
https://webs.iiitd.edu.in/raghava/hemopi2/download.html

HemoPI2:
https://webs.iiitd.edu.in/raghava/hemopi2/download/cross_val_dataset.csv
https://webs.iiitd.edu.in/raghava/hemopi2/download/independent_dataset.csv

Base de datos hemolitik, tiene API https://webs.iiitd.edu.in/raghava/hemolytik2/usr_guide.html#aboutapi    

# https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/main/pos.fa
"""

# Global variable
BASE_URL = "https://webs.iiitd.edu.in/raghava"
RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"
CURATED_FOLDER = "data/curated"
EXECUTION_DATE = datetime.now().strftime("%m_%Y")


def main():
    # # Downloading HemoPI data: 
    # # ---------------------------------------------------------------------
    # # 1️⃣ HemoPI – download FASTA files (pos/neg for each dataset & split)
    # # ---------------------------------------------------------------------

    # versions_hemopi = ["HemoPI_1_dataset", "HemoPI_2_dataset", "HemoPI_3_dataset"]
    # type_hemopi = ["main", "validation"]
    # labels_hemopi = {
    #     "positive": "pos",
    #     "negative": "neg"
    # }
    # output_hemopi = "HemoPI"
    # hemopi_path =  "hemopi/data"


    # tasks_hemopi =  generate_download_tasks(versions_hemopi, type_hemopi, labels_hemopi, RAW_FOLDER, BASE_URL, hemopi_path, output_hemopi, "fa")
    # for url, output_path in tasks_hemopi:
    #     print(f"📥 Downloading from: {url}")
    #     print(f"📁 Saving to: {output_path}")
    #     download_file_from_url(url, output_path)

    # # ---------------------------------------------------------------------
    # # 2️⃣ HemoPI2 – download two CSVs (cross‑val & independent)
    # # ---------------------------------------------------------------------
    # hemopi2_path = "hemopi2/download"
    # versions_hemopi2 = ["cross_val_dataset", "independent_dataset"]
    # output_hemopi2 = "HemoPI2"

    # tasks_hemopi2 =  generate_download_tasks(versions_hemopi2, None, None, RAW_FOLDER, BASE_URL, hemopi2_path, output_hemopi2, "csv")
    # for url, output_path in tasks_hemopi2:
    #     print(f"📥 Downloading from: {url}")
    #     print(f"📁 Saving to: {output_path}")
    #     download_file_from_url(url, output_path)


    # # ---------------------------------------------------------------------
    # # 2️⃣ HemoPI - Merge all HemoPi data
    # # ---------------------------------------------------------------------

    # raw_folder = Path("data/raw/HemoPI")
    # output_file = Path(f"data/processed/HemoPI/hemopi_all_{EXECUTION_DATE}.csv")
    # output_file.parent.mkdir(parents=True, exist_ok=True)

    # all_data = []

    # for fasta_file in raw_folder.glob("*.fa"):
    #     label = 1 if "positive" in fasta_file.name.lower() else 0
        
    #     # Leer secuencias del fasta
    #     for record in SeqIO.parse(fasta_file, "fasta"):
    #         seq = str(record.seq).upper()
    #         all_data.append({"SEQUENCE": seq, "label": label})

    # # Guardar en CSV
    # hemopi_all = pd.DataFrame(all_data)

    # # Clean data
    # hemopi_all_clean = clean_text_and_remove_duplicates(hemopi_all)
    # print(hemopi_all_clean.info())

    # hemopi_all_clean.to_csv(output_file, index=False)
    # print(f"📁 Saving to: {output_file}")


    # # ---------------------------------------------------------------------
    # # 3️⃣ Merge, clean, and save HemoPI2 data
    # # ---------------------------------------------------------------------

    # hemopi2_folder = Path(RAW_FOLDER)/output_hemopi2
    # output_file = Path(PROCESSED_FOLDER)/output_hemopi2/f"hemopi2_clean_{EXECUTION_DATE}.csv"
    
    # # Load and merge data
    # hemopi_2_all_df = load_and_concatenate_csvs(hemopi2_folder, versions_hemopi2)
    # print(hemopi_2_all_df.info())
    
    # # Clean data
    # hemopi_2_all_df_clean = clean_text_and_remove_duplicates(hemopi_2_all_df[['SEQUENCE','label']])
    # print(hemopi_2_all_df_clean.info())
    
    # # Save clean data
    # output_file.parent.mkdir(parents=True, exist_ok=True)
    # hemopi_2_all_df_clean.to_csv(output_file, index=False)
    # print(f"Cleaned data saved to {output_file}")
    # hemopi_2_all_df_clean = hemopi_2_all_df_clean

    # # ---------------------------------------------------------------------
    # # Merge HemoNet Data
    # # ---------------------------------------------------------------------

    # raw_folder = Path("data/raw/HemoNet")
    # output_file = Path(f"data/processed/HemoNet/hemonet_all_{EXECUTION_DATE}.csv")
    # output_file.parent.mkdir(parents=True, exist_ok=True)

    # all_data = []

    # VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

    # def clean_sequence(seq):
    #     return ''.join([c for c in seq.upper() if c in VALID_AA])

    # all_data = []

    # for fasta_file in raw_folder.glob("*.txt"):
    #     label = 0 if "nonhemo" in fasta_file.name.lower() else 1

    #     with open(fasta_file, encoding="utf-8", errors="ignore") as handle:
    #         for record in SeqIO.parse(handle, "fasta"):
    #             try:
    #                 seq = clean_sequence(str(record.seq))
    #                 if seq:  # solo guardar si no está vacía
    #                     all_data.append({"SEQUENCE": seq, "label": label})
    #             except Exception as e:
    #                 print(f"⚠️ Secuencia inválida en {fasta_file}, id={record.id}, error={e}")

    # # Guardar en CSV
    # hemonet_all = pd.DataFrame(all_data)

    # # Clean data
    # hemonet_all_clean = clean_text_and_remove_duplicates(hemonet_all)
    # print(hemonet_all_clean.info())


    # # 1. Identificar secuencias que aparecen con ambos labels
    # duplicadas = (
    #     hemonet_all_clean.groupby("SEQUENCE")["label"]
    #     .nunique()
    #     .reset_index()
    # )
    # duplicadas = duplicadas[duplicadas["label"] > 1]["SEQUENCE"]

    # # 2. Filtrar el dataframe eliminando esas secuencias
    # hemonet_filtrado = hemonet_all_clean[~hemonet_all_clean["SEQUENCE"].isin(duplicadas)]

    # hemonet_filtrado.reset_index(drop=True, inplace=True)
    # print(hemonet_filtrado.info())

    # hemonet_filtrado.to_csv(output_file, index=False)
    # print(f"📁 Saving to: {output_file}")
    
    # # ---------------------------------------------------------------------
    # # Merge HemoNet Data
    # # ---------------------------------------------------------------------

    # raw_folder = Path("data/raw/HemoDL")
    # output_file = Path(f"data/processed/HemoDL/hemodl_all_{EXECUTION_DATE}.csv")
    # output_file.parent.mkdir(parents=True, exist_ok=True)

    # all_data = []

    # for fasta_file in raw_folder.glob("*.fa"):
    #     label = 1 if "pos" in fasta_file.name.lower() else 0
        
    #     # Leer secuencias del fasta
    #     for record in SeqIO.parse(fasta_file, "fasta"):
    #         seq = str(record.seq).upper()
    #         all_data.append({"SEQUENCE": seq, "label": label})

    # # Guardar en CSV
    # hemodl_all = pd.DataFrame(all_data)

    # # Clean data
    # hemodl_all_clean = clean_text_and_remove_duplicates(hemodl_all)
    # print(hemodl_all_clean.info())

    # hemodl_all_clean.to_csv(output_file, index=False)
    # print(f"📁 Saving to: {output_file}")


    # # ---------------------------------------------------------------------
    # # Merge HLPPredFuse Data
    # # ---------------------------------------------------------------------

    # raw_folder = Path("data/raw/HLPPredFuse")
    # output_file = Path(f"data/processed/HLPPredFuse/hlppred_all_{EXECUTION_DATE}.csv")
    # output_file.parent.mkdir(parents=True, exist_ok=True)

    # all_data = []

    # for fasta_file in raw_folder.glob("*.txt"):
    #     label = 1 if "positive" in fasta_file.name.lower() else 0
        
    #     # Leer secuencias del fasta
    #     for record in SeqIO.parse(fasta_file, "fasta"):
    #         seq = str(record.seq).upper()
    #         all_data.append({"SEQUENCE": seq, "label": label})

    # # Guardar en CSV
    # hlppred_all = pd.DataFrame(all_data)

    # # Clean data
    # hlppred_all_clean = clean_text_and_remove_duplicates(hlppred_all)
    # print(hlppred_all_clean.info())

    # hlppred_all_clean.to_csv(output_file, index=False)
    # print(f"📁 Saving to: {output_file}")

    # # ---------------------------------------------------------------------
    # # Get complete data
    # # ---------------------------------------------------------------------
    # output_file = Path(PROCESSED_FOLDER) / f"dataset_master_{EXECUTION_DATE}.csv"
    # complete_hemopi_data = pd.concat([hemopi_all_clean, hemopi_2_all_df_clean, hemonet_filtrado, hemodl_all_clean, hlppred_all_clean])
    # complete_hemopi_data_clean = clean_text_and_remove_duplicates(complete_hemopi_data)
    # print(complete_hemopi_data_clean.info())
    # complete_hemopi_data_clean.to_csv(output_file, index=False)

    # # ---------------------------------------------------------------------
    # # analisis de duplicados con ambos label
    # # ---------------------------------------------------------------------

    # # 1. Identificar secuencias que aparecen con ambos labels
    # duplicadas = (
    #     complete_hemopi_data_clean.groupby("SEQUENCE")["label"]
    #     .nunique()
    #     .reset_index()
    # )
    # duplicadas = duplicadas[duplicadas["label"] > 1]["SEQUENCE"]
    # print(duplicadas)

    # # 2. Filtrar el dataframe eliminando esas secuencias
    # complete_hemopi_data_filter = complete_hemopi_data_clean[~complete_hemopi_data_clean["SEQUENCE"].isin(duplicadas)]

    # complete_hemopi_data_filter.reset_index(drop=True, inplace=True)
    # print(complete_hemopi_data_filter.info())

    # complete_hemopi_data_filter.to_csv(output_file, index=False)
    # print(f"📁 Saving to: {output_file}")

    # # ---------------------------------------------------------------------
    # # Datos basicos
    # # ---------------------------------------------------------------------

    # # Cantidad de aminoácidos en total (sumando longitudes de todas las secuencias)
    # total_aminoacids = complete_hemopi_data_filter["SEQUENCE"].str.len().sum()
    # total_secuencias = len(complete_hemopi_data_filter)

    # # Cantidad de ejemplos con label = 0
    # count_label_0 = (complete_hemopi_data_filter["label"] == 0).sum()

    # # Cantidad de ejemplos con label = 1
    # count_label_1 = (complete_hemopi_data_filter["label"] == 1).sum()

    # # Mostrar resultados
    # print("Total secuencias:", total_secuencias)
    # print("Total aminoácidos:", total_aminoacids)
    # print("Secuencias con label=0:", count_label_0)
    # print("Secuencias con label=1:", count_label_1)
    # avg_length = complete_hemopi_data_filter["SEQUENCE"].str.len().mean()
    # print("Longitud promedio de secuencias:", avg_length)




    # # ---------------------------------------------------------------------
    # # 4. Convert data in fasta
    # # ---------------------------------------------------------------------

    fasta_file_name = "peptides.fasta"
    fasta_path = f'{CURATED_FOLDER}/{fasta_file_name}'
    # df_to_fasta(complete_hemopi_data_filter, seq_col='SEQUENCE', output=fasta_path)

    # ---------------------------------------------------------------------
    # 5. Apply redundancy filtering using CD-HIT
    # ---------------------------------------------------------------------
    
    cd_hit_prefix = f"{CURATED_FOLDER}/cd_hit/peptides_nr"
    identities = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

    for identity in identities:
        run_cd_hit(fasta_path, cd_hit_prefix, identity)

        print(f"\nProcesando identidad {identity}...")

        fasta_file = f"{cd_hit_prefix}_{int(identity * 100)}.fasta"
        
        csv_output = f"{CURATED_FOLDER}/cd_hit_csv/peptides_{int(identity * 100)}.csv"
        data_identity = fasta_to_df(fasta_file, csv_output)
        print(data_identity)
        
    # ---------------------------------------------------------------------
    # 6. Calculate Amino Acid/Dimer/Trimer Frequencies
    # ---------------------------------------------------------------------

        if int(identity) == 1:
            print(f"\n Calculate aa frequencies")

            freq_df = _calculate_sequence_frequencies(data_identity, "SEQUENCE", relative=True)
            print(freq_df)

    # ---------------------------------------------------------------------
    # 6. Calculate Physicochemical Descriptors with Mordred
    # ---------------------------------------------------------------------
            
            #     # 5.1. Get SMILES 
            smile_folder = "smile_data"
            smiles_output = f"{CURATED_FOLDER}/{smile_folder}/peptides_{int(identity * 100)}_smiles.csv"
            # df_smiles = annotate_and_save(data_identity, output_csv=smiles_output)
            # print(df_smiles)

            df_smiles = pd.read_csv(smiles_output, delimiter=',')

            # # 3. Calcular descriptores Mordred
            df_desc = compute_mordred_descriptors(df_smiles)

            # 4. Guardar descriptores
            final_df = pd.concat([df_smiles, df_desc], axis=1)

            descriptor_folder= "descriptors"
            final_csv = f"{CURATED_FOLDER}/{descriptor_folder}/peptides_{int(identity * 100)}_descriptors.csv"
            final_df.to_csv(final_csv, index=False)

            #Merge descriptors + sequence-based frequencies + initial file
            hemo_path = Path(CURATED_FOLDER)/f"dataset_master_{EXECUTION_DATE}.csv"
            hemo_clean = pd.read_csv(hemo_path, delimiter=',')

            merge_all = (
                hemo_clean
                .merge(freq_df, how='inner', on='SEQUENCE')
                .merge(final_df, how='inner', on='SEQUENCE')
)
            final_path = f"{CURATED_FOLDER}/descriptors_and_frequencies/peptides_{int(identity * 100)}.csv"
            merge_all.to_csv(final_path, index=False)

            print(f"✅ saved {final_path}")
            df_100 = merge_all
            print(df_100.info())
        
        else:
            # obtener smile y descriptores y de sequence-based frequencies los otras identidades:
            merge_df = data_identity.merge(df_100, how='left' , on='SEQUENCE' )
            print(data_identity.info())
            final_csv = f"{CURATED_FOLDER}/descriptors_and_frequencies/peptides_{int(identity * 100)}.csv"
            merge_df.to_csv(final_csv, index=False)
            print(f"✅ saved {final_csv}")



def _calculate_sequence_frequencies(df: pd.DataFrame, seq_col: str, relative: bool = True) -> pd.DataFrame:
    """
    Computes mono-, di-, and tri-mer frequencies for each sequence in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a column with sequences.
        seq_col (str): Name of the column containing the sequences.
        relative (bool): If True, returns relative frequencies; if False, returns counts.

    Returns:
        pd.DataFrame: DataFrame with additional columns for each k-mer.
    """
    all_results = []
    
    for _, row in df.iterrows():
        seq = row[seq_col]
        
        mono_freq = calc_kmer_frequencies(seq, 1, relative)
        di_freq   = calc_kmer_frequencies(seq, 2, relative)
        tri_freq  = calc_kmer_frequencies(seq, 3, relative)
        
        # Unimos todo en un solo dict
        combined = {**mono_freq, **di_freq, **tri_freq}
        all_results.append(combined)
    
    # Convertimos a DataFrame y llenamos NaN con 0 (si un k-mer no aparece)
    freq_df = pd.DataFrame(all_results).fillna(0)
    
    # Unimos con las secuencias originales
    return pd.concat([df.reset_index(drop=True), freq_df], axis=1)


if __name__ == "__main__":
    main()


