import torch
import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Garante o reconhecimento da raiz para os imports[cite: 6]
sys.path.append(os.getcwd())

from dataset_utils import LPSD_Dataset
from models import create_vit_model, create_efficientnet_v2_model, create_resnet50_model, create_yolo11_cls_model, create_yolo11s_cls_model
from loss import FocalLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # se usar multi-GPU
    # Garante comportamento determinístico em operações do PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(split_folder, exp_name, model_type="effv2", freeze=True, augment=False, epochs=100, batch_size=32, lr=1e-4, patience=12, seed=42):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    set_seed(seed)

    # Descobre dinamicamente se é exp02, exp03 ou exp04 com base no caminho do split
    nome_experimento_pai = "checkpoints"
    for exp_id in ["exp02", "exp03", "exp04"]:
        if exp_id in split_folder.lower():
            nome_experimento_pai = f"checkpoints_{exp_id}"

    aug_suffix = "_withAug" if augment else "_noAug"
    suffix = f"{model_type}_frozen{aug_suffix}" if freeze else f"{model_type}_fine{aug_suffix}"
    full_exp_name = f"{exp_name}_seed{seed}_{suffix}"

    # Cria pastas separadas automaticamente: checkpoints_exp02/, checkpoints_exp03/, etc.
    save_path = f"{nome_experimento_pai}/{full_exp_name}"
    os.makedirs(save_path, exist_ok=True)

    last_ckpt_path = f"{save_path}/last_model.pth"
    best_ckpt_path = f"{save_path}/best_model.pth"
    log_path = f"{save_path}/train_progress.log"
    
    start_epoch, best_acc, epochs_no_improve = 0, 0.0, 0

    # 1. Carregamento dos Dados com Augmentation opcional[cite: 6]
    train_ds = LPSD_Dataset(f"{split_folder}/train.txt", augment=augment)
    val_ds = LPSD_Dataset(f"{split_folder}/val.txt", augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)

    # 2. Inicialização do Modelo
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()

    # Retomar treino se houver checkpoint[cite: 6]
    if os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

        # move os estados internos do otimizador Adam para a GPU atual
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    model = model.to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [{full_exp_name}]")
        
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

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).view(-1)
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        torch.cuda.empty_cache()
        val_acc = 100 * correct / total
        
        with open(log_path, "a") as f:
            f.write(f"Epocha {epoch+1}: Loss: {train_loss/len(train_loader):.4f} | Acc: {val_acc:.2f}%\n")

        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_acc': best_acc, 'epochs_no_improve': epochs_no_improve}
        torch.save(state, last_ckpt_path)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: break

    print(f"--- {full_exp_name} Finalizado! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_folder', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['vit', 'effv2', 'resnet50', 'yolo11', 'yolo11s'], default='effv2')
    parser.add_argument('--freeze', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--augment', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42, help='Semente aleatória para reprodutibilidade')
    args = parser.parse_args()
    train(
        split_folder=args.split_folder, 
        exp_name=args.exp_name, 
        model_type=args.model, 
        freeze=args.freeze, 
        augment=args.augment,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed
    )