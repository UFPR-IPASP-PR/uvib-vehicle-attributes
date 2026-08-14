import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())

from dataset_utils import LPSD_Dataset
from models import create_vit_model, create_efficientnet_v2_model, create_resnet50_model, create_yolo11s_cls_model
from loss import FocalLoss

def train(split_folder, exp_name, model_type="effv2", freeze=True, augment=False, epochs=100, batch_size=32, lr=1e-4, patience=12, seed=42):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mapeamento atualizado para utilizar o nome explicito das tarefas
    nome_experimento_pai = "checkpoints"
    for task_name in ["Orientation", "VMMRSuitability", "ColorClarity"]:
        if task_name.lower() in split_folder.lower():
            nome_experimento_pai = f"checkpoints_{task_name}"

    aug_suffix = "_withAug" if augment else "_noAug"
    suffix = f"{model_type}_frozen{aug_suffix}" if freeze else f"{model_type}_fine{aug_suffix}"
    full_exp_name = f"{exp_name}_seed{seed}_{suffix}"

    save_path = f"{nome_experimento_pai}/{full_exp_name}"
    os.makedirs(save_path, exist_ok=True)

    train_ds = LPSD_Dataset(f"{split_folder}/train.txt", augment=augment)
    val_ds = LPSD_Dataset(f"{split_folder}/val.txt", augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)

    models_map = {
        "effv2": create_efficientnet_v2_model,
        "vit": create_vit_model,
        "resnet50": create_resnet50_model,
        "yolo11s": create_yolo11s_cls_model
    }
    model = models_map[model_type](n_classes=2, freeze_backbone=freeze).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()

    best_acc, epochs_no_improve = 0.0, 0

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).view(-1)
            optimizer.zero_grad()
            outputs = model(imgs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).view(-1)
                outputs = model(imgs)
                if isinstance(outputs, tuple): outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break