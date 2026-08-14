import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import resnet50, ResNet50_Weights
#from ultralytics import YOLO

def create_vit_model(n_classes=2, freeze_backbone=True):
    # 1. carregar pesos pré-treinados (default = imagenet v1)
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # 2. congela as camadas (transfer learning)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 3. substitui cabeça de classificação
    # o ViT_B_16 tem uma camada de heads final
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, n_classes)

    # a nova camada 'heads.head' automaticamente tem requires_grand=True
    return model

def create_efficientnet_v2_model(n_classes=2, freeze_backbone=True):
    # cria o modelo com opção de congelamento
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    # lógica de congelamento (backbone congelado vs fine-tuning)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # substituição da cabeça de classificação
    # na efficientnet V2, a estrutura final é model.classifier, onde o índice [1]
    # é a camada Linear que faz a projeção para as classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)

    return model

def create_resnet50_model(n_classes=2, freeze_backbone=True):
    """
    Cria o modelo ResNet-50.
    A camada de classificação final da ResNet-50 chama-se 'fc'.
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Na ResNet-50, o model.fc é a camada linear final
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model

def create_yolo11_cls_model(n_classes=2, freeze_backbone=True):
    from ultralytics import YOLO
    # 1. Carrega o wrapper da Ultralytics
    model_yolo = YOLO('yolo11n-cls.pt')
    
    # 2. Extrai o modelo PyTorch puro (nn.Sequential)
    # No YOLO11-cls, o modelo real fica em model_yolo.model.model
    inner_model = model_yolo.model.model 

    if freeze_backbone:
        # Congela tudo
        for param in inner_model.parameters():
            param.requires_grad = False
            
    # 3. Acessa a cabeça de classificação (o último bloco da lista)
    # No YOLO11, o último bloco (índice -1) é a Classify head
    head_block = inner_model[-1]
    
    # 4. Dentro desse bloco, existe uma camada chamada 'linear'
    in_features = head_block.linear.in_features
    head_block.linear = torch.nn.Linear(in_features, n_classes)
    
    # Garante que a nova camada tenha gradiente ativo
    for param in head_block.linear.parameters():
        param.requires_grad = True

    return inner_model

def create_yolo11s_cls_model(n_classes=2, freeze_backbone=True):
    from ultralytics import YOLO
    # 1. Carrega o wrapper da Ultralytics
    model_yolo = YOLO('yolo11s-cls.pt')
    
    # 2. Extrai o modelo PyTorch puro (nn.Sequential)
    # No YOLO11-cls, o modelo real fica em model_yolo.model.model
    inner_model = model_yolo.model.model 

    if freeze_backbone:
        # Congela tudo
        for param in inner_model.parameters():
            param.requires_grad = False
            
    # 3. Acessa a cabeça de classificação (o último bloco da lista)
    # No YOLO11, o último bloco (índice -1) é a Classify head
    head_block = inner_model[-1]
    
    # 4. Dentro desse bloco, existe uma camada chamada 'linear'
    in_features = head_block.linear.in_features
    head_block.linear = torch.nn.Linear(in_features, n_classes)
    
    # Garante que a nova camada tenha gradiente ativo
    for param in head_block.linear.parameters():
        param.requires_grad = True

    return inner_model