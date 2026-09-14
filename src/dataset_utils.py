"""
Dataset utilities and data loading pipelines for the UVIB benchmark.

This module provides custom PyTorch Dataset implementations, augmentation pipelines
(including synthetic plate occlusion for VMMR suitability tests), and a balanced sampler 
for class-imbalanced binary classification.
"""

import os
import ast
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image


class RandomPlateOcclusion:
    """
    Applies synthetic occlusion noise over license plate boundaries or image regions.

    Args:
        p (float): Probability of applying occlusion. Defaults to 0.5.
        occlusion_types (list): Types of textures to apply ('black', 'white', 'noise', 'mean', 'blur').
    """

    def __init__(self, p=0.5, occlusion_types=None):
        self.p = p
        self.occlusion_types = occlusion_types or ['black', 'white', 'noise', 'mean', 'blur']

    def _generate_texture(self, h, w, c, occ_type, img_crop):
        if occ_type == 'black':
            return np.zeros((h, w, c), dtype=np.uint8)
        elif occ_type == 'white':
            return np.ones((h, w, c), dtype=np.uint8) * 255
        elif occ_type == 'mean':
            mean_color = img_crop.mean(axis=(0, 1)).astype(np.uint8)
            return np.full((h, w, c), mean_color, dtype=np.uint8)
        elif occ_type == 'noise':
            return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        elif occ_type == 'blur':
            return cv2.GaussianBlur(img_crop, (99, 99), 30)
        return np.zeros((h, w, c), dtype=np.uint8)

    def __call__(self, img_np, corners=None):
        if random.random() > self.p:
            return img_np, False

        h, w, c = img_np.shape
        occ_type = random.choice(self.occlusion_types)

        if corners and len(corners) == 4:
            pts = np.array(corners, dtype=np.float32)
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            plate_w = max_x - min_x

            mode = random.choice(['left', 'right', 'center_plate'])

            if mode == 'left':
                x1 = 0
                x2 = int(min_x + plate_w * 0.5)
            elif mode == 'right':
                x1 = int(max_x - plate_w * 0.5)
                x2 = w
            else:
                x1 = max(0, int(min_x - plate_w * 0.2))
                x2 = min(w, int(max_x + plate_w * 0.2))

            x1, x2 = max(0, x1), min(w, x2)

            if x2 > x1:
                img_np = img_np.copy()
                crop = img_np[:, x1:x2]
                texture = self._generate_texture(h, x2 - x1, c, occ_type, crop)
                img_np[:, x1:x2] = texture
                return img_np, True

        side = random.choice(['left', 'right'])
        mid_x = w // 2
        x1, x2 = (0, mid_x) if side == 'left' else (mid_x, w)

        img_np = img_np.copy()
        crop = img_np[:, x1:x2]
        texture = self._generate_texture(h, x2 - x1, c, occ_type, crop)
        img_np[:, x1:x2] = texture
        return img_np, True


def get_transforms(imgsz=224, augment=False):
    """
    Constructs Torchvision transform pipelines for training and evaluation.

    Args:
        imgsz (int): Input image resolution. Defaults to 224.
        augment (bool): If True, applies data augmentations (flips, rotations, jitter).

    Returns:
        transforms.Compose: PyTorch image transformation pipeline.
    """
    base_transforms = [
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ]
    if augment:
        augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(augmentation + base_transforms)
    return transforms.Compose(base_transforms)


class UVIBDataset(Dataset):
    """
    PyTorch Dataset implementation for loading UVIB benchmark partitions.

    Args:
        txt_path (str): Path to the split file (.txt).
        data_root (str): Root directory containing image datasets. Defaults to "data".
        imgsz (int): Target image resolution for model input. Defaults to 224.
        augment (bool): Whether to enable training data augmentations.
        use_plate_occlusion (bool): Whether to enable dynamic synthetic occlusion.
    """

    def __init__(self, txt_path, data_root="data", imgsz=224, augment=False, use_plate_occlusion=False):
        self.samples = []
        self.data_root = data_root
        self.imgsz = imgsz
        self.augment = augment
        self.use_plate_occlusion = use_plate_occlusion and augment
        
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Split text file not found: {txt_path}")

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    crop_path = parts[0]
                    label = int(parts[1])
                    corners = None
                    plate = None

                    if len(parts) >= 3 and parts[2] != "None":
                        try:
                            corners = ast.literal_eval(parts[2])
                        except Exception:
                            corners = None
                    if len(parts) >= 4:
                        plate = parts[3]

                    self.samples.append({
                        'crop_path': crop_path,
                        'label': label,
                        'corners': corners,
                        'plate': plate
                    })
        
        self.transform = get_transforms(imgsz=imgsz, augment=augment)
        self.plate_occlusion = RandomPlateOcclusion(p=0.5)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        rel_path = item['crop_path']
        label = item['label']
        corners = item['corners']

        full_path = os.path.join(self.data_root, rel_path)

        try:
            image = Image.open(full_path).convert('RGB')
            img_np = np.array(image)

            if self.use_plate_occlusion and label == 0:
                img_np, ocluded = self.plate_occlusion(img_np, corners)
                if ocluded:
                    label = 1

            image = Image.fromarray(img_np)
            image = self.transform(image)
        except Exception:
            # Fallback tensor in case of corrupted or missing images
            image = torch.zeros((3, self.imgsz, self.imgsz))
            
        return image, torch.tensor(label, dtype=torch.long)


class BalancedSampler(Sampler):
    """
    Custom Sampler for balancing mini-batch class distributions during training.

    Args:
        dataset (Dataset): UVIBDataset instance.
        batch_size (int): Mini-batch size.
        n_classes (int): Number of target classes.
    """

    def __init__(self, dataset, batch_size, n_classes):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.indices = self._make_indices()

    def _make_indices(self):
        label_to_indices = {i: [] for i in range(self.n_classes)}
        for idx, item in enumerate(self.dataset.samples):
            label_to_indices[item['label']].append(idx)
        return label_to_indices

    def __iter__(self):
        ret = []
        per_class = self.batch_size // self.n_classes
        for _ in range(len(self.dataset) // self.batch_size):
            for c in range(self.n_classes):
                if len(self.indices[c]) > 0:
                    ret.extend(np.random.choice(self.indices[c], per_class))
        return iter(ret)

    def __len__(self):
        return len(self.dataset)