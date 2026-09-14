"""
Model factory module for the UVIB benchmark architectures.

This module provides builder functions for instantiating and adapting deep learning 
backbones (ViT-B/16, EfficientNet-V2-S, ResNet-50, YOLO11n-cls, and YOLO11s-cls)
pre-trained on ImageNet for binary classification tasks.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights
)


def create_vit_model(n_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Creates and configures a Vision Transformer (ViT-B/16) model.

    Args:
        n_classes (int): Number of target output classes. Defaults to 2.
        freeze_backbone (bool): Whether to freeze pre-trained backbone parameters. Defaults to True.

    Returns:
        nn.Module: Adapted PyTorch ViT model.
    """
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, n_classes)

    return model


def create_efficientnet_v2_model(n_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Creates and configures an EfficientNet-V2-Small model.

    Args:
        n_classes (int): Number of target output classes. Defaults to 2.
        freeze_backbone (bool): Whether to freeze pre-trained backbone parameters. Defaults to True.

    Returns:
        nn.Module: Adapted PyTorch EfficientNet-V2 model.
    """
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)

    return model


def create_resnet50_model(n_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Creates and configures a ResNet-50 model.

    Args:
        n_classes (int): Number of target output classes. Defaults to 2.
        freeze_backbone (bool): Whether to freeze pre-trained backbone parameters. Defaults to True.

    Returns:
        nn.Module: Adapted PyTorch ResNet-50 model.
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model


def create_yolo11_cls_model(n_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Creates and configures a YOLO11-Nano classification model (yolo11n-cls).

    Args:
        n_classes (int): Number of target output classes. Defaults to 2.
        freeze_backbone (bool): Whether to freeze pre-trained backbone parameters. Defaults to True.

    Returns:
        nn.Module: Extracted native PyTorch neural network module.
    """
    from ultralytics import YOLO

    model_yolo = YOLO('yolo11n-cls.pt')
    inner_model = model_yolo.model.model

    if freeze_backbone:
        for param in inner_model.parameters():
            param.requires_grad = False

    head_block = inner_model[-1]
    in_features = head_block.linear.in_features
    head_block.linear = nn.Linear(in_features, n_classes)

    for param in head_block.linear.parameters():
        param.requires_grad = True

    return inner_model


def create_yolo11s_cls_model(n_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Creates and configures a YOLO11-Small classification model (yolo11s-cls).

    Args:
        n_classes (int): Number of target output classes. Defaults to 2.
        freeze_backbone (bool): Whether to freeze pre-trained backbone parameters. Defaults to True.

    Returns:
        nn.Module: Extracted native PyTorch neural network module.
    """
    from ultralytics import YOLO

    model_yolo = YOLO('yolo11s-cls.pt')
    inner_model = model_yolo.model.model

    if freeze_backbone:
        for param in inner_model.parameters():
            param.requires_grad = False

    head_block = inner_model[-1]
    in_features = head_block.linear.in_features
    head_block.linear = nn.Linear(in_features, n_classes)

    for param in head_block.linear.parameters():
        param.requires_grad = True

    return inner_model