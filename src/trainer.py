"""
Training pipeline for the UVIB benchmark experiments.

This script executes the model training loop using Adam optimizer (lr=1e-4),
Focal Loss, Early Stopping based on Macro F1 validation metric, and automatic
checkpointing for fine-tuning or feature extraction evaluations.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dataset_utils import UVIBDataset
from models import (
    create_vit_model, 
    create_efficientnet_v2_model, 
    create_resnet50_model, 
    create_yolo11s_cls_model
)
from loss import FocalLoss


def set_seed(seed: int):
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(split_folder, exp_name, data_root="data", model_type="effv2", freeze=True, 
          augment=False, epochs=100, batch_size=32, lr=1e-4, patience=12, seed=42):
    """
    Executes model training across a benchmark protocol split.

    Args:
        split_folder (str): Directory path containing train.txt and val.txt.
        exp_name (str): Protocol experiment identifier (e.g. S2G, G2S, All, CDS).
        data_root (str): Path to root directory containing source images.
        model_type (str): Network architecture ('effv2', 'vit', 'resnet50', 'yolo11s').
        freeze (bool): Whether to freeze backbone parameters.
        augment (bool): Whether to apply data augmentations.
        epochs (int): Maximum number of training epochs. Defaults to 100.
        batch_size (int): Mini-batch size. Defaults to 32.
        lr (float): Learning rate for Adam optimizer. Defaults to 1e-4.
        patience (int): Early stopping patience epochs. Defaults to 12.
        seed (int): Random seed. Defaults to 42.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    set_seed(seed)

    # Infer parent checkpoint folder name
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

    save_path = os.path.join(nome_experimento_pai, full_exp_name)
    os.makedirs(save_path, exist_ok=True)

    last_ckpt_path = os.path.join(save_path, "last_model.pth")
    best_ckpt_path = os.path.join(save_path, "best_model.pth")
    log_path = os.path.join(save_path, "train_progress.log")
    
    start_epoch, best_f1, epochs_no_improve = 0, 0.0, 0

    # Load datasets
    train_txt = os.path.join(split_folder, "train.txt")
    val_txt = os.path.join(split_folder, "val.txt")

    train_ds = UVIBDataset(train_txt, data_root=data_root, augment=augment)
    val_ds = UVIBDataset(val_txt, data_root=data_root, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=False)

    # Model instantiation
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()

    # Resume from checkpoint if available
    if os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    model = model.to(device)

    print(f"[*] Starting training for experiment: {full_exp_name}")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{full_exp_name}]")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).view(-1)
            optimizer.zero_grad()
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).view(-1)
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        torch.cuda.empty_cache()
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)
        
        # Log progress
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Macro F1: {val_f1:.4f}\n")

        # Save current state
        state = {
            'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'best_f1': best_f1, 
            'epochs_no_improve': epochs_no_improve
        }
        torch.save(state, last_ckpt_path)

        # Early stopping logic based on Macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            epochs_no_improve = 0
            print(f"   ↳ Best Model Saved! New Val Macro F1: {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   ↳ Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"✅ [SUCCESS] Experiment {full_exp_name} finished! Best Val Macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer script for UVIB benchmark.")
    parser.add_argument('--split_folder', type=str, required=True, help="Path to split directory containing train.txt/val.txt")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment identifier (e.g. S2G, G2S, All, CDS)")
    parser.add_argument('--data_root', type=str, default="data", help="Root directory containing images")
    parser.add_argument('--model', type=str, choices=['vit', 'effv2', 'resnet50', 'yolo11', 'yolo11s'], default='effv2')
    parser.add_argument('--freeze', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--augment', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train(
        split_folder=args.split_folder, 
        exp_name=args.exp_name, 
        data_root=args.data_root,
        model_type=args.model, 
        freeze=args.freeze, 
        augment=args.augment,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed
    )