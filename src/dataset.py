import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_DIR,
    REPORTS_DIR,
    CLASS_TO_IDX,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG_SAMPLE_SIZE,
)
from src.preprocessing import get_train_transforms, get_eval_transforms

class AgriFreshDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for AgriFresh.
    Images are loaded from disk on-the-fly to minimize RAM usage.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.file_paths = self.df["file_path"].tolist()
        self.freshness_labels = self.df["freshness"].map(CLASS_TO_IDX).tolist()
        self.produces = self.df["produce"].tolist() if "produce" in self.df.columns else ["Unknown"] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.freshness_labels[idx]
        produce = self.produces[idx]

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            # Fallback for corrupted image (safety net)
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), produce, img_path

def create_stratified_splits(metadata_csv_path: Path = REPORTS_DIR / "image_level_metadata.csv", force_recreate: bool = False):
    """
    Creates reproducible, stratified train / val / test splits (70 / 15 / 15).
    Stratification is performed on both Freshness and Produce type.
    """
    train_csv = DATASET_DIR / "train_split.csv"
    val_csv = DATASET_DIR / "val_split.csv"
    test_csv = DATASET_DIR / "test_split.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists() and not force_recreate:
        print(f"Loading existing split manifests from {DATASET_DIR}...")
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        df_test = pd.read_csv(test_csv)
        return df_train, df_val, df_test

    if not metadata_csv_path.exists():
        from src.dataset_analysis import inspect_dataset
        inspect_dataset()

    df = pd.read_csv(metadata_csv_path)
    valid_df = df[~df["is_corrupt"]].copy()

    # Create joint stratification key (e.g., 'Fresh_Banana', 'Rotten_Tomato')
    valid_df["strat_key"] = valid_df["freshness"].astype(str) + "_" + valid_df["produce"].astype(str)

    # 1. Split into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        valid_df,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=valid_df["strat_key"],
        random_state=RANDOM_SEED,
    )

    # 2. Split Temp (30%) equally into Val (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["strat_key"],
        random_state=RANDOM_SEED,
    )

    # Clean up and save
    train_df = train_df.drop(columns=["strat_key"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["strat_key"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strat_key"]).reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Created stratified splits:")
    print(f"  - Train: {len(train_df):,} images ({len(train_df)/len(valid_df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df):,} images ({len(val_df)/len(valid_df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df):,} images ({len(test_df)/len(valid_df)*100:.1f}%)")

    return train_df, val_df, test_df

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    debug_mode: bool = False,
    debug_samples: int = DEBUG_SAMPLE_SIZE,
):
    """
    Returns PyTorch DataLoaders for train, val, and test sets.
    """
    df_train, df_val, df_test = create_stratified_splits()

    if debug_mode:
        # Sample uniformly across freshness classes
        train_samples = min(len(df_train), int(debug_samples * 0.70))
        val_samples = min(len(df_val), int(debug_samples * 0.15))
        test_samples = min(len(df_test), int(debug_samples * 0.15))

        df_train = df_train.sample(n=train_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        df_val = df_val.sample(n=val_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        df_test = df_test.sample(n=test_samples, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"Debug subsets: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    train_dataset = AgriFreshDataset(df_train, transform=get_train_transforms())
    val_dataset = AgriFreshDataset(df_val, transform=get_eval_transforms())
    test_dataset = AgriFreshDataset(df_test, transform=get_eval_transforms())

    use_pin = torch.cuda.is_available()
    use_persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )

    return train_loader, val_loader, test_loader, df_train, df_val, df_test

if __name__ == "__main__":
    train_df, val_df, test_df = create_stratified_splits(force_recreate=True)
    t_loader, v_loader, te_loader, _, _, _ = get_dataloaders(debug_mode=True)
    for images, labels, produces, paths in t_loader:
        print(f"Sample Batch Shape: Images={images.shape}, Labels={labels.shape}")
        break
