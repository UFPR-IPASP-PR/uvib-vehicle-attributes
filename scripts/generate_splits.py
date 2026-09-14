"""
Script to generate split text files for the UVIB benchmark protocols from UVIBv1.json.

This script parses the master annotation JSON file (UVIBv1.json) and generates 
stratified split text files (.txt) for four evaluation protocols across the three binary tasks:
  1. Surveillance-to-General (S2G)
  2. General-to-Surveillance (G2S)
  3. All-Datasets (All)
  4. Cross-Dataset Shift (CDS)

Directory Structure Generated:
    splits/
    ├── splits_expColorClarity/
    │   ├── split_All/
    │   ├── split_CDS/
    │   ├── split_G2S/
    │   └── split_S2G/
    ├── splits_expOrientation/
    │   └── ...
    └── splits_expVMMRSuitability/
        └── ...

Output format per line in generated text files:
    <Dataset_Name>/<Image_Name> \t <Binary_Label>

Target Binary Label Mapping:
    - Orientation: 0 = Front | 1 = Rear
    - VMMR Suitability: 0 = Suitable | 1 = Unsuitable
    - Color Clarity: 0 = Color | 1 = Non-Color
"""

import os
import json
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths relative to repository structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

JSON_MASTER_PATH = os.path.join(PROJECT_ROOT, 'UVIBv1.json')

# Explicit output directories aligned with experiment naming
RAIZ_COLOR = os.path.join(PROJECT_ROOT, 'splits', 'splits_expColorClarity')
RAIZ_ORIENTATION = os.path.join(PROJECT_ROOT, 'splits', 'splits_expOrientation')
RAIZ_SUITABILITY = os.path.join(PROJECT_ROOT, 'splits', 'splits_expVMMRSuitability')

# Known dataset directories for reliable relative path extraction
KNOWN_DATASETS = [
    'UFPR-VeSV', 'LPLCv2', 'Selected-Vehicle-Rear', 
    'UFPR-ALPR', 'RodoSol-ALPR', 'SSIG-SegPlate', 'UFOP'
]


def salvar_txt_split(df_particao, coluna_label, caminho_arquivo):
    """
    Saves partition data into a text file using relative paths: Dataset/image_name.jpg label_id.

    Args:
        df_particao (pd.DataFrame): Dataframe partition containing image paths and labels.
        coluna_label (str): Name of the target label column.
        caminho_arquivo (str): Output text file path.

    Returns:
        int: Total number of samples saved.
    """
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    
    with open(caminho_arquivo, "w") as f:
        for _, row in df_particao.iterrows():
            caminho_completo = str(row['image_path']).strip()
            label_val = int(row[coluna_label])
            
            # Normalize path separators across OS platforms
            caminho_normalizado = caminho_completo.replace('\\', '/')
            partes = caminho_normalizado.split('/')
            
            # Dynamically extract dataset name and filename
            dataset_nome = None
            for p in partes:
                if p in KNOWN_DATASETS:
                    dataset_nome = p
                    break
            
            if dataset_nome:
                nome_imagem = partes[-1]
                caminho_relativo = f"{dataset_nome}/{nome_imagem}"
            else:
                caminho_relativo = f"{partes[-2]}/{partes[-1]}" if len(partes) >= 2 else caminho_completo

            f.write(f"{caminho_relativo}\t{label_val}\n")
            
    return len(df_particao)


def processar_protocolos_para_label(df_dados, coluna_label, raiz_saida):
    """
    Generates dataset splits across all four evaluation protocols for a target attribute.

    Args:
        df_dados (pd.DataFrame): Dataframe loaded from annotations.
        coluna_label (str): Column name representing the task binary label.
        raiz_saida (str): Destination directory for the experiment splits.
    """
    print(f"\n[⚙️] Generating splits for task: {coluna_label} -> Output directory: {raiz_saida}")
    
    if os.path.exists(raiz_saida):
        shutil.rmtree(raiz_saida)
    os.makedirs(raiz_saida, exist_ok=True)
    
    # Domain masks based on source datasets
    is_vesv = df_dados['image_path'].str.contains('UFPR-VeSV', na=False)
    is_lplc = df_dados['image_path'].str.contains('LPLCv2', na=False)
    is_rear = df_dados['image_path'].str.contains('Selected-Vehicle-Rear', na=False)
    
    is_ufpr_alpr = df_dados['image_path'].str.contains('UFPR-ALPR', na=False)
    is_rodosol = df_dados['image_path'].str.contains('RodoSol-ALPR', na=False)
    is_ssig = df_dados['image_path'].str.contains('SSIG-SegPlate', na=False)
    is_ufop = df_dados['image_path'].str.contains('UFOP', na=False)
    
    # ---------------------------------------------------------
    # PROTOCOL 1: Surveillance-to-General (S2G)
    # ---------------------------------------------------------
    df_surv = df_dados[is_vesv | is_lplc | is_rear].copy()
    df_gen = df_dados[is_ufpr_alpr | is_rodosol | is_ssig | is_ufop].copy()
    
    s2g_train, s2g_val = train_test_split(
        df_surv, test_size=0.40, random_state=42, stratify=df_surv[coluna_label]
    )
    
    p_s2g = os.path.join(raiz_saida, "split_S2G")
    n_tr = salvar_txt_split(s2g_train, coluna_label, os.path.join(p_s2g, "train.txt"))
    n_va = salvar_txt_split(s2g_val, coluna_label, os.path.join(p_s2g, "val.txt"))
    n_te = salvar_txt_split(df_gen, coluna_label, os.path.join(p_s2g, "test.txt"))
    print(f"   ↳ Protocol S2G completed -> Train: {n_tr} | Val: {n_va} | Test: {n_te}")

    # ---------------------------------------------------------
    # PROTOCOL 2: General-to-Surveillance (G2S)
    # ---------------------------------------------------------
    g2s_train, g2s_val = train_test_split(
        df_gen, test_size=0.40, random_state=42, stratify=df_gen[coluna_label]
    )
    
    p_g2s = os.path.join(raiz_saida, "split_G2S")
    n_tr = salvar_txt_split(g2s_train, coluna_label, os.path.join(p_g2s, "train.txt"))
    n_va = salvar_txt_split(g2s_val, coluna_label, os.path.join(p_g2s, "val.txt"))
    n_te = salvar_txt_split(df_surv, coluna_label, os.path.join(p_g2s, "test.txt"))
    print(f"   ↳ Protocol G2S completed -> Train: {n_tr} | Val: {n_va} | Test: {n_te}")

    # ---------------------------------------------------------
    # PROTOCOL 3: All-Datasets (All)
    # ---------------------------------------------------------
    all_train, all_temp = train_test_split(
        df_dados, test_size=0.40, random_state=42, stratify=df_dados[coluna_label]
    )
    all_val, all_test = train_test_split(
        all_temp, test_size=0.50, random_state=42, stratify=all_temp[coluna_label]
    )
    
    p_all = os.path.join(raiz_saida, "split_All")
    n_tr = salvar_txt_split(all_train, coluna_label, os.path.join(p_all, "train.txt"))
    n_va = salvar_txt_split(all_val, coluna_label, os.path.join(p_all, "val.txt"))
    n_te = salvar_txt_split(all_test, coluna_label, os.path.join(p_all, "test.txt"))
    print(f"   ↳ Protocol All completed -> Train: {n_tr} | Val: {n_va} | Test: {n_te}")

    # ---------------------------------------------------------
    # PROTOCOL 4: Cross-Dataset Shift (CDS)
    # ---------------------------------------------------------
    df_cds_dev = df_dados[is_vesv | is_rodosol | is_rear | is_ufop].copy()
    df_cds_test = df_dados[is_lplc | is_ufpr_alpr | is_ssig].copy()
    
    cds_train, cds_val = train_test_split(
        df_cds_dev, test_size=0.40, random_state=42, stratify=df_cds_dev[coluna_label]
    )
    
    p_cds = os.path.join(raiz_saida, "split_CDS")
    n_tr = salvar_txt_split(cds_train, coluna_label, os.path.join(p_cds, "train.txt"))
    n_va = salvar_txt_split(cds_val, coluna_label, os.path.join(p_cds, "val.txt"))
    n_te = salvar_txt_split(df_cds_test, coluna_label, os.path.join(p_cds, "test.txt"))
    print(f"   ↳ Protocol CDS completed -> Train: {n_tr} | Val: {n_va} | Test: {n_te}")


if __name__ == "__main__":
    print(f"[*] Reading master annotations from: {JSON_MASTER_PATH}")

    if not os.path.exists(JSON_MASTER_PATH):
        raise FileNotFoundError(f"Critical Error: {JSON_MASTER_PATH} not found!")

    with open(JSON_MASTER_PATH, 'r') as f:
        dados_json = json.load(f)

    # Convert JSON structure to DataFrame for split calculations
    df_completo = pd.DataFrame(dados_json)

    # Process each target binary task
    processar_protocolos_para_label(df_completo, 'label_color', RAIZ_COLOR)
    processar_protocolos_para_label(df_completo, 'label_orientation', RAIZ_ORIENTATION)
    processar_protocolos_para_label(df_completo, 'label_suitability', RAIZ_SUITABILITY)

    print("\n" + "="*70)
    print("🎉 [SUCCESS] All protocol splits generated successfully from UVIBv1.json!")
    print("="*70)