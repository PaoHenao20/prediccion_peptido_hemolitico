import pandas as pd
from extract import download_file_from_url
from transform import annotate_and_save, clean_df, compute_mordred_descriptors, concat_csv_sin_duplicados, get_smiles

# HemoPI2 data - information link: https://webs.iiitd.edu.in/raghava/hemopi2/download.html
# ToxinPred3 - information link:  https://webs.iiitd.edu.in/raghava/toxinpred3/download.php

MAIN_FOLDER = "data/raw"

url_dict =  {
    "HemoPI_train": "https://webs.iiitd.edu.in/raghava/hemopi2/download/cross_val_dataset.csv",
    "HemoPI_test": "https://webs.iiitd.edu.in/raghava/hemopi2/download/independent_dataset.csv",
#     # "ToxinPred_train_pos": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/train_pos.csv",
#     # "ToxinPred_train_neg": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/train_neg.csv",
#     # "ToxinPred_test_pos": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/test_pos.csv",
#     # "ToxinPred_test_neg": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/test_neg.csv"

}

if __name__ == "__main__":
    for file_name, url in url_dict.items():
        download_file_from_url(url, f"{MAIN_FOLDER}/{file_name}.csv")



    salida = "data/processed/all_data.csv"

    resultado = concat_csv_sin_duplicados(MAIN_FOLDER,
                                            pattern="*.csv",
                                            output_path=salida)
    print(f"Concatenados {len(resultado)} filas únicas.")


    # Limpia y guarda
    df_limpio = clean_df(
        resultado,
        output_csv="data/processed/all_data_clean.csv"
    )

    print(df_limpio.info())


    df_result = annotate_and_save(
        df_limpio,
        seq_col="SEQUENCE",
        smiles_col="SMILES",
        output_csv="data/processed/all_data_with_smiles.csv"
    )
    print(df_result.head())




    df_orig = pd.read_csv("data/processed/all_data_with_smiles.csv")
    # df_orig = df_orig.iloc[[0],:]
    # print(df_orig)

    # Calculo descriptores
    df_mordred = compute_mordred_descriptors(df_orig, smiles_col='SMILES')

    # Uno con el original (sin perder columnas)
    df_final = pd.concat([df_orig, df_mordred], axis=1)

    # Guardo el resultado
    df_final.to_csv("data/processed/all_data_with_descriptor.csv", index=False)

    print("Descriptores calculados:", df_mordred.shape[1])
    print(df_final.head())