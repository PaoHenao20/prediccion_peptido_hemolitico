import os
from pathlib import Path
from venv import logger

import pandas as pd
from data_processing.extractor import download_file_from_url
from data_processing.transformation import (
    generate_download_tasks,
    clean_text_and_remove_duplicates,
    load_and_concatenate_csvs,
    run_cd_hit)

# url Reference" https://webs.iiitd.edu.in/raghava/hemopi/datasets.php
# https://webs.iiitd.edu.in/raghava/hemopi2/download.html

#https://webs.iiitd.edu.in/raghava/hemopi2/download/cross_val_dataset.csv
# https://webs.iiitd.edu.in/raghava/hemopi2/download/independent_dataset.csv

#Base de datos hemolitik, tiene API https://webs.iiitd.edu.in/raghava/hemolytik2/usr_guide.html#aboutapi    

# https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/main/pos.fa


# Global variable
BASE_URL = "https://webs.iiitd.edu.in/raghava"
RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"


def main():
    # Downloading HemoPI data: 
    # ---------------------------------------------------------------------
    # 1️⃣ HemoPI – download FASTA files (pos/neg for each dataset & split)
    # ---------------------------------------------------------------------

    versions_hemopi = ["HemoPI_1_dataset", "HemoPI_2_dataset", "HemoPI_3_dataset"]
    type_hemopi = ["main", "validation"]
    labels_hemopi = {
        "positive": "pos",
        "negative": "neg"
    }
    output_hemopi = "HemoPI"
    hemopi_path =  "hemopi/data"


    tasks_hemopi =  generate_download_tasks(versions_hemopi, type_hemopi, labels_hemopi, RAW_FOLDER, BASE_URL, hemopi_path, output_hemopi, "fa")
    for url, output_path in tasks_hemopi:
        print(f"📥 Downloading from: {url}")
        print(f"📁 Saving to: {output_path}")
        download_file_from_url(url, output_path)

    # ---------------------------------------------------------------------
    # 2️⃣ HemoPI2 – download two CSVs (cross‑val & independent)
    # ---------------------------------------------------------------------
    hemopi2_path = "hemopi2/download"
    versions_hemopi2 = ["cross_val_dataset", "independent_dataset"]
    output_hemopi2 = "HemoPI2"

    tasks_hemopi2 =  generate_download_tasks(versions_hemopi2, None, None, RAW_FOLDER, BASE_URL, hemopi2_path, output_hemopi2, "csv")
    for url, output_path in tasks_hemopi2:
        print(f"📥 Downloading from: {url}")
        print(f"📁 Saving to: {output_path}")
        download_file_from_url(url, output_path)


    # ---------------------------------------------------------------------
    # 3️⃣ Merge, clean, and save HemoPI2 data
    # ---------------------------------------------------------------------

    hemopi2_folder = Path(RAW_FOLDER)/output_hemopi2
    output_file = Path(PROCESSED_FOLDER)/output_hemopi2/"hemopi2_clean.csv"
    
    # Load and merge data
    hemopi_2_all_df = load_and_concatenate_csvs(hemopi2_folder, versions_hemopi2)
    print(hemopi_2_all_df.info())
    
    # Clean data
    hemopi_2_all_df_clean = clean_text_and_remove_duplicates(hemopi_2_all_df)
    print(hemopi_2_all_df_clean.info())
    
    # Save clean data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hemopi_2_all_df_clean.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")



if __name__ == "__main__":
    main()


