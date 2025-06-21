# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%
import numpy as np
import pandas as pd
from extract import download_file_from_url
from transform import annotate_and_save, clean_df, compute_mordred_descriptors, concat_csv_sin_duplicados, get_smiles, df_to_fasta, run_cd_hit, fasta_to_df, compute_sequence_features, plot_top_correlations, compute_statistical_summary_by_class, t_test_between_classes, remove_highly_correlated_features
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore", message="Failed to patch pandas")

# Asegúrate de que las carpetas existen
os.makedirs("data/curated", exist_ok=True)
os.makedirs("data/raw", exist_ok=True) 
os.makedirs("data/curated/cd_hit", exist_ok=True)
os.makedirs("data/curated/cd_hit_csv", exist_ok=True)
os.makedirs("data/curated/descriptors", exist_ok=True)
os.makedirs("data/curated/descriptors_and_frequencies", exist_ok=True)
os.makedirs("data/curated/smile_data", exist_ok=True)
os.makedirs("data/result/figures", exist_ok=True)

# %% [markdown]
# HemoPI2 data - information link: https://webs.iiitd.edu.in/raghava/hemopi2/download.html
# ToxinPred3 - information link:  https://webs.iiitd.edu.in/raghava/toxinpred3/download.php

# %%
MAIN_FOLDER = "data/raw_test"
CURATED_FOLDER = "data/curated"

# %%
url_dict =  {
    "HemoPI_train": "https://webs.iiitd.edu.in/raghava/hemopi2/download/cross_val_dataset.csv",
    "HemoPI_test": "https://webs.iiitd.edu.in/raghava/hemopi2/download/independent_dataset.csv",
    }

# %%
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

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


    df_without_cd_hit = df_limpio.merge(df_100, how='left' , on='SEQUENCE' )
    df_without_cd_hit_mod = df_without_cd_hit.rename(columns={"μM": "HC50"})
    path = f"{CURATED_FOLDER}/descriptors_and_frequencies/peptides_without_cd_hit.csv"
    df_without_cd_hit_mod.to_csv(path, index=False)
    print(f"✅ saved {path}")

    # %%
    """1. Remove low-variance and missing-value variables:
    From the full dataset without CD-HIT filtering, drop all variables with missing values.
    Additionally, remove variables (descriptors, dimer, and trimer frequencies) with a standard deviation less than 0.05, as they likely contain values close to zero and contribute little to model performance.
    This filtering step is independent of class labels."""
    # %%
    # Quitar columnas con valores faltantes

    df_without_cd_hit_mod = df_without_cd_hit_mod.dropna(how='any')
    df_clean = df_without_cd_hit_mod.dropna(axis=1)

    # nan_counts = df_without_cd_hit_mod.isna().sum()
    # nan_cols = nan_counts[nan_counts > 0]
    # print(f"Columnas con NaNs: {len(nan_cols)}")
    # print(nan_cols.sort_values(ascending=False).head(10)) 

    # Separar variable objetivo
    target_cols = ['label', 'HC50']
    features = df_clean.drop(columns=target_cols, errors='ignore')

    # Filtrar solo columnas numéricas
    numeric_features = features.select_dtypes(include=[np.number])

    # Quitar columnas con desviación estándar < 0.05
    low_var_cols = numeric_features.std()[numeric_features.std() < 0.05].index
    features_filtered = numeric_features.drop(columns=low_var_cols)

    # Dataset final con columnas útiles + target
    df_filtered = pd.concat([features_filtered, df_clean[target_cols]], axis=1)

    print(df_filtered)
    # %%
    # Preparar los datos para visualización
    features_only = df_filtered.drop(columns=['label', 'HC50'], errors='ignore')
    features_only['label'] = df_filtered['label']

    # Agrupar las columnas en grupos de 50
    cols = features_only.drop(columns='label').columns
    group_size = 50
    column_groups = [cols[i:i+group_size] for i in range(0, len(cols), group_size)]

    output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'TFM', 'prediccion_peptido_hemolitico', 'data', 'result', 'figures'))
    os.makedirs(output_dir, exist_ok=True)



    # Crear los boxplots
    for i, group in enumerate(column_groups, start=1):
        df_plot = pd.melt(features_only, id_vars='label', value_vars=group,
                        var_name='feature', value_name='value')
        
        plt.figure(figsize=(24, 10))
        sns.boxplot(data=df_plot, x='feature', y='value', hue='label')
        plt.title(f'Boxplots por clase – Variables {((i-1)*group_size)+1} a {((i-1)*group_size)+len(group)}')
        plt.xticks(rotation=90)
        plt.legend(title='Clase', labels=['No Hemolítico (0)', 'Hemolítico (1)'])
        plt.tight_layout()
        filepath = os.path.join(output_dir, f'boxplot_group_{i}.png')
        plt.savefig(filepath)
        plt.close()

    # %%
    target_variables = ['label', 'HC50']

    # Crear directorio para guardar las figuras
    output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'TFM', 'prediccion_peptido_hemolitico', 'data', 'result', 'correlation'))
    os.makedirs(output_dir, exist_ok=True)

    for method in ['pearson', 'spearman']:
        plot_top_correlations(df_filtered, method, target_variables, output_dir)

    
    output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'TFM', 'prediccion_peptido_hemolitico', 'data', 'result', 'statistical_summary_by_class'))
    os.makedirs(output_dir, exist_ok=True)

    summary = compute_statistical_summary_by_class(df_filtered, class_col='label', exclude_cols=['HC50'], output_dir=output_dir)

    # Mostrar o exportar
    summary.head()


    t_test_results = t_test_between_classes(df_filtered, class_col='label', exclude_cols=['HC50'])

    # Guardar o visualizar
    t_test_results.to_csv("data/result/statistical_summary_by_class/t_test_by_class.csv", index=False)
    t_test_results.head(10)


    # Por ejemplo, para eliminar con Spearman y 0.8 de umbral
    df_filtered_corr, dropped_vars = remove_highly_correlated_features(df_filtered, method='spearman', threshold=0.8, exclude_cols=['label', 'HC50'])
    df_filtered_corr.to_csv("data/result/ remove_highly_correlated.csv", index=False)






    # %%


