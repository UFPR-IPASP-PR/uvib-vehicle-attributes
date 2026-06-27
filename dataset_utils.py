import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def get_transforms(imgsz=224, augment=False):
    """
    Centraliza as transformações. 
    O tamanho 224 é mantido por ser o padrão de pré-treino da EfficientNetV2-S.
    """
    base_transforms = [
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ]
    
    if augment:
        # Adiciona variações para robustez nos experimentos 02 e 03
        augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(augmentation + base_transforms)
    
    return transforms.Compose(base_transforms)

class LPSD_Dataset(Dataset):
    def __init__(self, txt_path, imgsz=224, augment=False):
        """
        txt_path: Caminho para o arquivo .txt dos splits.
        imgsz: Tamanho da imagem (224 para manter compatibilidade com pesos pré-treinados).
        augment: Se True, aplica data augmentation (usar apenas no treino).
        """
        self.samples = []
        self.imgsz = imgsz
        
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {txt_path}")

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.samples.append((parts[0], int(parts[1])))
        
        # Define as transformações usando a função utilitária
        self.transform = get_transforms(imgsz=imgsz, augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Em caso de erro, retorna um tensor de zeros para não interromper a fila
            image = torch.zeros((3, self.imgsz, self.imgsz))
            
        return image, torch.tensor([label], dtype=torch.long)

class BalancedSampler(Sampler):
    """
    Mantém o equilíbrio entre classes (ex: Suitable vs Unsuitable),
    ajudando a mitigar o recall baixo em classes minoritárias.
    """
    def __init__(self, dataset, batch_size, n_classes):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.indices = self._make_indices()

    def _make_indices(self):
        label_to_indices = {i: [] for i in range(self.n_classes)}
        for idx, (_, label) in enumerate(self.dataset.samples):
            label_to_indices[label].append(idx)
        return label_to_indices

    def __iter__(self):
        ret = []
        # Calcula quantos itens de cada classe por batch
        per_class = self.batch_size // self.n_classes
        for _ in range(len(self.dataset) // self.batch_size):
            for c in range(self.n_classes):
                if len(self.indices[c]) > 0:
                    ret.extend(np.random.choice(self.indices[c], per_class))
        return iter(ret)

    def __len__(self):
        return len(self.dataset)