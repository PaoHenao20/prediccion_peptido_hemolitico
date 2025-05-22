from extract import download_file_from_url
from transform import concat_csv_sin_duplicados, get_smiles

# HemoPI2 data - information link: https://webs.iiitd.edu.in/raghava/hemopi2/download.html
# ToxinPred3 - information link:  https://webs.iiitd.edu.in/raghava/toxinpred3/download.php

MAIN_FOLDER = "data/raw"

url_dict =  {
    "HemoPI_train": "https://webs.iiitd.edu.in/raghava/hemopi2/download/cross_val_dataset.csv",
    "HemoPI_test": "https://webs.iiitd.edu.in/raghava/hemopi2/download/independent_dataset.csv",
    # "ToxinPred_train_pos": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/train_pos.csv",
    # "ToxinPred_train_neg": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/train_neg.csv",
    # "ToxinPred_test_pos": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/test_pos.csv",
    # "ToxinPred_test_neg": "https://webs.iiitd.edu.in/raghava/toxinpred3/download/test_neg.csv"

}

for file_name, url in url_dict.items():
    download_file_from_url(url, f"{MAIN_FOLDER}/{file_name}.csv")



salida = "data/processed/all_data.csv"

resultado = concat_csv_sin_duplicados(MAIN_FOLDER,
                                        pattern="*.csv",
                                        output_path=salida)
print(f"Concatenados {len(resultado)} filas únicas.")

# seq_input = "PYK-K-W-P-R-P-D-A-P-I-P-P"
# smiles = get_smiles(seq_input)
# print("SMILES generado:", smiles)