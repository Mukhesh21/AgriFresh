import torchvision.transforms as transforms
from PIL import Image
from src.config import IMAGE_SIZE

# Standard ImageNet normalization values for pretrained CNN models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size=IMAGE_SIZE):
    """
    Enhanced augmentation pipeline for training:
    - RandomResizedCrop for scale invariance
    - Random Horizontal & Vertical Flip
    - Random Rotation (+/- 20 degrees)
    - Aggressive Color Jitter for lighting/color variation
    - Random Perspective for viewpoint invariance
    - Tensor conversion & ImageNet normalization
    - Random Erasing for occlusion robustness
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])

def get_eval_transforms(image_size=IMAGE_SIZE):
    """
    Deterministic preprocessing pipeline for Validation and Testing:
    - Resize to standard input size (224x224)
    - Strictly NO random augmentations
    - Tensor conversion & ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_inference_transform(image_size=IMAGE_SIZE):
    """Transform for single-image real-time inference."""
    return get_eval_transforms(image_size=image_size)
