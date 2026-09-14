"""
Evaluation and inference script for the UVIB benchmark protocols.

This script evaluates trained models on designated test splits across all four benchmark 
protocols (S2G, G2S, All, CDS) and target tasks, outputting quantitative metrics 
(Macro F1, Precision, Recall, Confusion Matrices) and logging misclassified samples.
"""

import os
import sys
import argparse
import random
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models import (
    create_vit_model, 
    create_efficientnet_v2_model, 
    create_resnet50_model, 
    create_yolo11s_cls_model
)
from dataset_utils import UVIBDataset


def set_seed(seed: int):
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_task_classes(split_folder: str):
    """
    Determines class names based on target task inferred from split path.

    Returns:
        tuple: (dict mapping index to name, list of class names in order)
    """
    folder_lower = split_folder.lower()
    if 'color' in folder_lower:
        class_map = {0: "Color", 1: "Non-Color"}
    elif 'suitability' in folder_lower:
        class_map = {0: "Suitable", 1: "Unsuitable"}
    else:
        # Default fallback to Orientation
        class_map = {0: "Front", 1: "Rear"}

    target_names = [class_map[0], class_map[1]]
    return class_map, target_names


def evaluate(split_folder, exp_name, data_root="data", model_type='effv2', freeze=True, augment=False, seed=42):
    """
    Evaluates a trained model checkpoint on the test set of a specified split.

    Args:
        split_folder (str): Directory containing split files (e.g. splits/splits_expOrientation/split_S2G).
        exp_name (str): Protocol experiment name (e.g. S2G, G2S, All, CDS).
        data_root (str): Root directory for source image datasets.
        model_type (str): Network backbone architecture ('effv2', 'vit', 'resnet50', 'yolo11s').
        freeze (bool): Whether backbone weights were frozen during training.
        augment (bool): Whether training used augmentations.
        seed (int): Random seed for reproducibility.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    set_seed(seed)

    # Determine experiment parent checkpoint folder
    nome_experimento_pai = "checkpoints"
    folder_lower = split_folder.lower()
    if "color" in folder_lower:
        nome_experimento_pai = "checkpoints_expColorClarity"
    elif "orientation" in folder_lower:
        nome_experimento_pai = "checkpoints_expOrientation"
    elif "suitability" in folder_lower:
        nome_experimento_pai = "checkpoints_expVMMRSuitability"

    aug_suffix = "_withAug" if augment else "_noAug"
    suffix = f"{model_type}_frozen{aug_suffix}" if freeze else f"{model_type}_fine{aug_suffix}"
    full_exp_name = f"{exp_name}_seed{seed}_{suffix}"

    model_path = os.path.join(nome_experimento_pai, full_exp_name, "best_model.pth")
    output_path = os.path.join(nome_experimento_pai, full_exp_name, "results")
    error_dir = os.path.join(nome_experimento_pai, full_exp_name, "error_images")
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    test_txt = os.path.join(split_folder, "test.txt")
    test_ds = UVIBDataset(test_txt, data_root=data_root, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Instantiate model backbone
    if model_type == "effv2":
        model = create_efficientnet_v2_model(n_classes=2, freeze_backbone=freeze)
    elif model_type == "vit":
        model = create_vit_model(n_classes=2, freeze_backbone=freeze)
    elif model_type == "resnet50":
        model = create_resnet50_model(n_classes=2, freeze_backbone=freeze)
    elif model_type in ["yolo11", "yolo11s"]:
        model = create_yolo11s_cls_model(n_classes=2, freeze_backbone=freeze)
    else:
        raise ValueError(f"Unrecognized model architecture: {model_type}")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    classes, target_names = get_task_classes(split_folder)

    print(f"[*] Evaluating Model: {full_exp_name}")
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs, labels = imgs.to(device), labels.to(device).view(-1)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            
            p, l = predicted.item(), labels.item()
            all_preds.append(p)
            all_labels.append(l)

            # Log misclassified samples
            if p != l:
                src_path = os.path.join(data_root, test_ds.samples[i]['crop_path'])
                target_dir = os.path.join(error_dir, f"Real_{classes.get(l, l)}_Pred_{classes.get(p, p)}")
                os.makedirs(target_dir, exist_ok=True)
                dst_path = os.path.join(target_dir, os.path.basename(src_path))

                if not os.path.exists(dst_path) and os.path.exists(src_path):
                    try:
                        os.symlink(src_path, dst_path)
                    except Exception:
                        pass  # Handle systems without symlink permissions

    # Metrics calculation and reporting
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Normalized Confusion Matrix - {full_exp_name}")
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    
    metrics_file = os.path.join(output_path, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Evaluation Results for Experiment: {full_exp_name}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        
    print(f"✅ [SUCCESS] Evaluation completed. Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for UVIB benchmark.")
    parser.add_argument('--split_folder', type=str, required=True, help="Path to split directory containing test.txt")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment identifier (e.g. S2G, G2S, All, CDS)")
    parser.add_argument('--data_root', type=str, default="data", help="Root directory containing images")
    parser.add_argument('--model', type=str, choices=['vit', 'effv2', 'resnet50', 'yolo11', 'yolo11s'], default='effv2')
    parser.add_argument('--freeze', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--augment', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    evaluate(args.split_folder, args.exp_name, args.data_root, args.model, args.freeze, args.augment, args.seed)