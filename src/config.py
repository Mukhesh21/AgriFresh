import os
from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw Dataset Location
# The dataset has nested directories from extraction
DEFAULT_RAW_DATASET_PATH = (
    PROJECT_ROOT
    / "AgriFreshNET Freshness and Shelf-Life Image Datase"
    / "AgriFreshNET Freshness and Shelf-Life Image Datase"
    / "Processed Data"
    / "Processed Data"
)

# Project Output Directories
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
METRICS_DIR = RESULTS_DIR / "model_metrics"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
TRAINING_HISTORY_DIR = RESULTS_DIR / "training_history"

# Create directories if they do not exist
for d in [
    DATASET_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    METRICS_DIR,
    CONFUSION_MATRICES_DIR,
    TRAINING_HISTORY_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Freshness Classification (Core Task)
CLASS_NAMES = ["Fresh", "Semi-Fresh", "Rotten"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 3

# Produce Types Present in AgriFreshNET
PRODUCE_TYPES = [
    "Banana",
    "Bittermelon",
    "Cucumber",
    "Eggplant",
    "Orange",
    "Papaya",
    "Pineapple",
    "Tomato",
]

# Dataset Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Configurations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64       # Larger batch keeps RTX 3050 GPU fed efficiently
NUM_WORKERS = 2       # 2 parallel CPU workers to prevent Windows shared memory limits
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4

# Debug Mode Settings
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 120  # Fast verification on small subset
DEBUG_EPOCHS = 2

# Models to Train and Compare
MODEL_NAMES = [
    "mobilenetv3_small",
    "efficientnet_b0",
    "resnet18",
    "densenet121",
    "efficientnet_v2_s",
]

MODEL_DISPLAY_NAMES = {
    "mobilenetv3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
    "densenet121": "DenseNet121",
    "efficientnet_v2_s": "EfficientNetV2-S",
}
