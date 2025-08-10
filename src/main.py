import os
from pathlib import Path
from venv import logger

import pandas as pd
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
    # # 3️⃣ Merge, clean, and save HemoPI2 data
    # # ---------------------------------------------------------------------

    # hemopi2_folder = Path(RAW_FOLDER)/output_hemopi2
    # output_file = Path(PROCESSED_FOLDER)/output_hemopi2/"hemopi2_clean.csv"
    
    # # Load and merge data
    # hemopi_2_all_df = load_and_concatenate_csvs(hemopi2_folder, versions_hemopi2)
    # print(hemopi_2_all_df.info())
    
    # # Clean data
    # hemopi_2_all_df_clean = clean_text_and_remove_duplicates(hemopi_2_all_df)
    # print(hemopi_2_all_df_clean.info())
    
    # # Save clean data
    # output_file.parent.mkdir(parents=True, exist_ok=True)
    # hemopi_2_all_df_clean.to_csv(output_file, index=False)
    # print(f"Cleaned data saved to {output_file}")


    # hemopi_2_all_df_clean = hemopi_2_all_df_clean.head(100)

    # # ---------------------------------------------------------------------
    # # 4. Convert data in fasta
    # # ---------------------------------------------------------------------

    fasta_file_name = "peptides.fasta"
    fasta_path = f'{CURATED_FOLDER}/{fasta_file_name}'
    # df_to_fasta(hemopi_2_all_df_clean, seq_col='SEQUENCE', output=fasta_path)

    # # ---------------------------------------------------------------------
    # # 5. Apply redundancy filtering using CD-HIT
    # # ---------------------------------------------------------------------
    
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
            
                # 5.1. Get SMILES 
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
            output_hemopi2 = "HemoPI2"
            hemopi_path = Path(PROCESSED_FOLDER)/output_hemopi2/"hemopi2_clean.csv"
            hemopi2_clean = pd.read_csv(hemopi_path, delimiter=',')

            merge_all = (
                hemopi2_clean
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



    # df_without_cd_hit = df_limpio.merge(df_100, how='left' , on='SEQUENCE' )
    # df_without_cd_hit_mod = df_without_cd_hit.rename(columns={"μM": "HC50"})
    # path = f"{CURATED_FOLDER}/descriptors_and_frequencies/peptides_without_cd_hit.csv"
    # df_without_cd_hit_mod.to_csv(path, index=False)
    # print(f"✅ saved {path}")




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


