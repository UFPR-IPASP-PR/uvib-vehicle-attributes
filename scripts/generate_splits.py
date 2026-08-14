import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_MASTER_PATH = os.path.join(SCRIPT_DIR, 'auditoria/dataset_multitask_unificado.csv')

# Novas pastas nomeadas com clareza
RAIZ_ORIENTATION = os.path.join(SCRIPT_DIR, 'splits/Orientation')
RAIZ_SUITABILITY = os.path.join(SCRIPT_DIR, 'splits/VMMRSuitability')
RAIZ_COLOR = os.path.join(SCRIPT_DIR, 'splits/ColorClarity')

def salvar_txt_split(df_particao, coluna_label, caminho_arquivo):
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    with open(caminho_arquivo, "w") as f:
        for _, row in df_particao.iterrows():
            img_path = str(row['image_path']).strip()
            label_val = int(row[coluna_label])
            f.write(f"{img_path} {label_val}\n")
    return len(df_particao)

def processar_protocolos_para_label(df_dados, coluna_label, raiz_saida):
    if os.path.exists(raiz_saida):
        shutil.rmtree(raiz_saida)
    os.makedirs(raiz_saida, exist_ok=True)
    
    is_vesv = df_dados['image_path'].str.contains('UFPR-VeSV', na=False)
    is_lplc = df_dados['image_path'].str.contains('LPLCv2', na=False)
    is_rear = df_dados['image_path'].str.contains('Selected-Vehicle-Rear', na=False)
    
    is_ufpr_alpr = df_dados['image_path'].str.contains('UFPR-ALPR', na=False)
    is_rodosol = df_dados['image_path'].str.contains('RodoSol-ALPR', na=False)
    is_ssig = df_dados['image_path'].str.contains('SSIG-SegPlate', na=False)
    is_ufop = df_dados['image_path'].str.contains('UFOP', na=False)
    
    # 1. S2G
    df_surv = df_dados[is_vesv | is_lplc | is_rear].copy()
    df_gen = df_dados[is_ufpr_alpr | is_rodosol | is_ssig | is_ufop].copy()
    s2g_tr, s2g_va = train_test_split(df_surv, test_size=0.40, random_state=42, stratify=df_surv[coluna_label])
    p_s2g = os.path.join(raiz_saida, "split_S2G")
    salvar_txt_split(s2g_tr, coluna_label, os.path.join(p_s2g, "train.txt"))
    salvar_txt_split(s2g_va, coluna_label, os.path.join(p_s2g, "val.txt"))
    salvar_txt_split(df_gen, coluna_label, os.path.join(p_s2g, "test.txt"))

    # 2. G2S
    g2s_tr, g2s_va = train_test_split(df_gen, test_size=0.40, random_state=42, stratify=df_gen[coluna_label])
    p_g2s = os.path.join(raiz_saida, "split_G2S")
    salvar_txt_split(g2s_tr, coluna_label, os.path.join(p_g2s, "train.txt"))
    salvar_txt_split(g2s_va, coluna_label, os.path.join(p_g2s, "val.txt"))
    salvar_txt_split(df_surv, coluna_label, os.path.join(p_g2s, "test.txt"))

    # 3. All
    all_tr, all_temp = train_test_split(df_dados, test_size=0.40, random_state=42, stratify=df_dados[coluna_label])
    all_va, all_te = train_test_split(all_temp, test_size=0.50, random_state=42, stratify=all_temp[coluna_label])
    p_all = os.path.join(raiz_saida, "split_All")
    salvar_txt_split(all_tr, coluna_label, os.path.join(p_all, "train.txt"))
    salvar_txt_split(all_va, coluna_label, os.path.join(p_all, "val.txt"))
    salvar_txt_split(all_te, coluna_label, os.path.join(p_all, "test.txt"))

    # 4. CDS
    df_cds_dev = df_dados[is_vesv | is_rodosol | is_rear | is_ufop].copy()
    df_cds_test = df_dados[is_lplc | is_ufpr_alpr | is_ssig].copy()
    cds_tr, cds_va = train_test_split(df_cds_dev, test_size=0.40, random_state=42, stratify=df_cds_dev[coluna_label])
    p_cds = os.path.join(raiz_saida, "split_CDS")
    salvar_txt_split(cds_tr, coluna_label, os.path.join(p_cds, "train.txt"))
    salvar_txt_split(cds_va, coluna_label, os.path.join(p_cds, "val.txt"))
    salvar_txt_split(df_cds_test, coluna_label, os.path.join(p_cds, "test.txt"))

if __name__ == "__main__":
    df_completo = pd.read_csv(CSV_MASTER_PATH)
    processar_protocolos_para_label(df_completo, 'label_orientation', RAIZ_ORIENTATION)
    processar_protocolos_para_label(df_completo, 'label_suitability', RAIZ_SUITABILITY)
    processar_protocolos_para_label(df_completo, 'label_color', RAIZ_COLOR)