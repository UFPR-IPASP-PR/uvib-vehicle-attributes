import os
import sys
import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset_utils import LPSD_Dataset
from models import create_vit_model, create_efficientnet_v2_model, create_resnet50_model, create_yolo11s_cls_model

# Mapeamento dos rótulos de cada tarefa para o Classification Report
TASK_CLASSES = {
    "Orientation": ["Front", "Rear"],
    "VMMRSuitability": ["Suitable", "Unsuitable"],
    "ColorClarity": ["Color", "Non-Color"]
}

def evaluate(split_folder, exp_name, model_type='effv2', freeze=False, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    task_identified = "Orientation"
    for task in TASK_CLASSES.keys():
        if task.lower() in split_folder.lower():
            task_identified = task
            break

    full_exp_name = f"{exp_name}_seed{seed}_{model_type}_fine_noAug"
    checkpoint_dir = f"checkpoints_{task_identified}/{full_exp_name}"
    
    test_ds = LPSD_Dataset(f"{split_folder}/test.txt", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    models_map = {
        "effv2": create_efficientnet_v2_model,
        "vit": create_vit_model,
        "resnet50": create_resnet50_model,
        "yolo11s": create_yolo11s_cls_model
    }
    model = models_map[model_type](n_classes=2, freeze_backbone=freeze)
    model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device).view(-1)
            outputs = model(imgs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.item())
            all_labels.append(labels.item())

    target_names = TASK_CLASSES[task_identified]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    
    os.makedirs(f"{checkpoint_dir}/results", exist_ok=True)
    with open(f"{checkpoint_dir}/results/metrics_direction.txt", "w") as f:
        f.write(report)