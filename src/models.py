import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NUM_CLASSES

def build_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Builds one of the 5 deep learning models with transfer learning weights
    and customized 3-class classification head (Fresh, Semi-Fresh, Rotten).
    """
    model_name_clean = model_name.lower().replace("-", "_").strip()

    if model_name_clean in ["mobilenetv3_small", "mobilenetv3", "mobilenet_v3_small"]:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name_clean in ["efficientnet_b0", "efficientnetb0"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name_clean in ["resnet18", "resnet_18"]:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name_clean in ["densenet121", "densenet_121"]:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name_clean in ["efficientnet_v2_s", "efficientnetv2_s", "efficientnetv2"]:
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models are: "
            "mobilenetv3_small, efficientnet_b0, resnet18, densenet121, efficientnet_v2_s"
        )

    return model

def freeze_backbone(model: nn.Module, model_name: str):
    """
    Stage 1: Freezes the pretrained feature extractor backbone and keeps only the classification head trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    model_name_clean = model_name.lower().replace("-", "_").strip()
    if model_name_clean in ["resnet18", "resnet_18"]:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name_clean in ["densenet121", "densenet_121"]:
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True

def unfreeze_all(model: nn.Module):
    """
    Stage 2: Unfreezes all parameters for end-to-end fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True

def count_parameters(model: nn.Module):
    """
    Returns (total_params, trainable_params).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_model_size_mb(model: nn.Module):
    """
    Calculates estimated in-memory model parameter size in Megabytes.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(size_all_mb, 2)

if __name__ == "__main__":
    from src.config import MODEL_NAMES, MODEL_DISPLAY_NAMES
    print("=" * 60)
    print("FIVE DEEP LEARNING MODEL ARCHITECTURE PROFILING")
    print("=" * 60)
    for mname in MODEL_NAMES:
        model = build_model(mname, pretrained=False)
        total, trainable = count_parameters(model)
        size_mb = calculate_model_size_mb(model)
        print(f"[{MODEL_DISPLAY_NAMES[mname]}]")
        print(f"  Total Params:     {total:,}")
        print(f"  Trainable Params: {trainable:,}")
        print(f"  Model Size (MB):  {size_mb} MB\n")
