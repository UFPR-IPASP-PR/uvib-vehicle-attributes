import torch
import os
import sys
import argparse
import random
import numpy as np
import torch
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Correção de Path para evitar ModuleNotFoundError[cite: 5]
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models import create_vit_model, create_efficientnet_v2_model, create_resnet50_model, create_yolo11_cls_model, create_yolo11s_cls_model
    
from dataset_utils import LPSD_Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # se usar multi-GPU
    # Garante comportamento determinístico em operações do PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(split_folder, exp_name, model_type='effv2', freeze=True, augment=False, seed=42):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    set_seed(seed)

    # Descobre dinamicamente de qual pasta de checkpoints carregar
    nome_experimento_pai = "checkpoints"
    for exp_id in ["exp02", "exp03", "exp04"]:
        if exp_id in split_folder.lower():
            nome_experimento_pai = f"checkpoints_{exp_id}"

    aug_suffix = "_withAug" if augment else "_noAug"
    suffix = f"{model_type}_frozen{aug_suffix}" if freeze else f"{model_type}_fine{aug_suffix}"
    full_exp_name = f"{exp_name}_seed{seed}_{suffix}"

    model_path = f"{nome_experimento_pai}/{full_exp_name}/best_model.pth"
    output_path = f"{nome_experimento_pai}/{full_exp_name}/results"
    error_dir = f"{nome_experimento_pai}/{full_exp_name}/error_images"
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Avaliação sempre sem augmentation[cite: 5]
    test_ds = LPSD_Dataset(f"{split_folder}/test.txt", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    if model_type == "effv2":
        model = create_efficientnet_v2_model(n_classes=2, freeze_backbone=freeze)
    elif model_type == "vit":
        model = create_vit_model(n_classes=2, freeze_backbone=freeze)
    elif model_type == "resnet50":
        model = create_resnet50_model(n_classes=2, freeze_backbone=freeze)
    elif model_type == "yolo11":
        model = create_yolo11_cls_model(n_classes=2, freeze_backbone=freeze)
    elif model_type == "yolo11s":
        model = create_yolo11s_cls_model(n_classes=2, freeze_backbone=freeze)
    else:
        raise ValueError(f"Modelo {model_type} não reconhecido!")

    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    classes = {0: "Frontal", 1: "Traseira"} # Mapeamento específico do Exp 03[cite: 5]

    print(f"[*] Avaliando Direção: {full_exp_name}")
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device).view(-1)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            
            p, l = predicted.item(), labels.item()
            all_preds.append(p)
            all_labels.append(l)

            if p != l:
                src_path = test_ds.samples[i][0]
                target_dir = os.path.join(error_dir, f"Real_{classes[l]}_Pred_{classes[p]}")
                os.makedirs(target_dir, exist_ok=True)
                dst_path = os.path.join(target_dir, os.path.basename(src_path))

                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)

    target_names = ['Frontal', 'Traseira']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
    plt.savefig(f"{output_path}/direction_confusion_matrix.png")
    plt.close()
    
    with open(f"{output_path}/metrics_direction.txt", "w") as f:
        f.write(report)
    print(f"Avaliação concluída. Resultados em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_folder', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    # Adicionamos 'choices' para o script aceitar as strings 'vit' e 'resnet50'
    parser.add_argument('--model', type=str, choices=['vit', 'effv2', 'resnet50', 'yolo11', 'yolo11s'], default='effv2')
    parser.add_argument('--freeze', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--augment', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--seed', type=int, default=42, help='Semente aleatória para reprodutibilidade')
    args = parser.parse_args()
    evaluate(args.split_folder, args.exp_name, args.model, args.freeze, args.augment, args.seed)