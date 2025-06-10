import pandas as pd
from extract import download_file_from_url
from transform import annotate_and_save, clean_df, compute_mordred_descriptors, concat_csv_sin_duplicados, get_smiles, df_to_fasta, run_cd_hit, fasta_to_df, compute_sequence_features

# HemoPI2 data - information link: https://webs.iiitd.edu.in/raghava/hemopi2/download.html
# ToxinPred3 - information link:  https://webs.iiitd.edu.in/raghava/toxinpred3/download.php

MAIN_FOLDER = "data/raw_test"
CURATED_FOLDER = "data/curated"

url_dict =  {
    "HemoPI_train": "https://webs.iiitd.edu.in/raghava/hemopi2/download/cross_val_dataset.csv",
    "HemoPI_test": "https://webs.iiitd.edu.in/raghava/hemopi2/download/independent_dataset.csv",

}

if __name__ == "__main__":
    for file_name, url in url_dict.items():
        download_file_from_url(url, f"{MAIN_FOLDER}/{file_name}.csv")

    all_data_file_name = "all_data.csv"

    resultado = concat_csv_sin_duplicados(MAIN_FOLDER,
                                            pattern="*.csv",
                                            output_path=f"{CURATED_FOLDER}/{all_data_file_name}")
    print(f"Concatenados {len(resultado)} filas únicas.")

    data_clean_name = "all_data_clean.csv"
    # clean and save
    df_limpio = clean_df(
        resultado,
        output_csv=f"{CURATED_FOLDER}/{data_clean_name}"
    )

    print(df_limpio.info())
    fasta_file_name = "peptides.fasta"
    fasta_path = f'{CURATED_FOLDER}/{fasta_file_name}'
    fasta = df_to_fasta(df_limpio, seq_col='SEQUENCE', output=fasta_path)

    # path = f"{CURATED_FOLDER}"
    cd_hit_prefix = f"{CURATED_FOLDER}/cd_hit/peptides_nr"
    smile_folder = "smile_data"
    descriptor_folder= "descriptors"
    identities = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

    for identity in identities:
        run_cd_hit(fasta_path, cd_hit_prefix, identity)

        print(f"\nProcesando identidad {identity}...")

        fasta_file = f"{cd_hit_prefix}_{int(identity * 100)}.fasta"
        
        csv_output = f"{CURATED_FOLDER}/cd_hit_csv/peptides_{int(identity * 100)}.csv"
        df = fasta_to_df(fasta_file, csv_output)


        #Get SMILES and descriptors:
        if int(identity) == 1:
            # Calcular features
            features_df = compute_sequence_features(df, seq_column='SEQUENCE')
            print(features_df.head())
            
             # 2. Get SMILES 
            smiles_output = f"{CURATED_FOLDER}/{smile_folder}/peptides_{int(identity * 100)}_smiles.csv"
            df_smiles = annotate_and_save(df, output_csv=smiles_output)

            # 3. Calcular descriptores Mordred
            df_desc = compute_mordred_descriptors(df_smiles)

            # 4. Guardar descriptores
            final_df = pd.concat([df_smiles, df_desc], axis=1)
            final_csv = f"{CURATED_FOLDER}/{descriptor_folder}/peptides_{int(identity * 100)}_descriptors.csv"
            final_df.to_csv(final_csv, index=False)

            #Merge descriptors + sequence-based frequencies
            merge_all = final_df.merge(features_df, how='inner', on='SEQUENCE' )
            final_path = f"{CURATED_FOLDER}/descriptors_and_frequencies/peptides_{int(identity * 100)}.csv"
            merge_all.to_csv(final_path, index=False)

            print(f"✅ saved {final_path}")
            df_100 = merge_all
            print(df_100.info())
        
        else:
            # obtener smile y descriptores y de sequence-based frequencies los otras identidades:
            merge_df = df.merge(df_100, how='left' , on='SEQUENCE' )
            print(df.info())
            final_csv = f"{CURATED_FOLDER}/descriptors_and_frequencies/peptides_{int(identity * 100)}.csv"
            merge_df.to_csv(final_csv, index=False)
            print(f"✅ saved {final_csv}")

        