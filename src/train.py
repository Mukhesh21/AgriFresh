import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_NAMES,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    TRAINING_HISTORY_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    DEBUG_MODE,
    DEBUG_EPOCHS,
    RANDOM_SEED,
)
from src.dataset import get_dataloaders
from src.models import build_model, freeze_backbone, unfreeze_all, count_parameters
from src.utils import plot_training_history

# Set CPU thread count for maximum parallel processing performance
if hasattr(os, "cpu_count") and os.cpu_count():
    torch.set_num_threads(os.cpu_count())

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def train_single_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and device.type == "cuda"

    for images, labels, _, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return val_loss, val_acc

def train_model(
    model_name: str,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    debug_mode: bool = DEBUG_MODE,
    device: str = None,
):
    """
    Trains one model using 2-stage transfer learning with early stopping and checkpointing.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"TRAINING: {display_name} (Device: {device}, Debug: {debug_mode})")
    print("=" * 70)

    # 1. Load Data
    train_loader, val_loader, _, _, _, _ = get_dataloaders(
        batch_size=batch_size,
        debug_mode=debug_mode,
    )

    # 2. Build Model
    model = build_model(model_name, pretrained=True)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model Parameters: Total = {total_params:,} | Trainable = {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    # AMP GradScaler for mixed precision (GPU only)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    if scaler:
        print(f"  [AMP] Automatic Mixed Precision enabled for faster GPU training.")

    # Stage 1: Freeze backbone, train head
    freeze_backbone(model, model_name)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    start_time = time.time()
    unfreeze_epoch = 2 if epochs > 3 else 1

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Stage 2 Transition: Unfreeze full network with smaller LR
        if epoch == unfreeze_epoch + 1:
            print(f"\n---> [Stage 2] Unfreezing full backbone for fine-tuning at lower learning rate...")
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.2, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        train_loss, train_acc = train_single_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "lr": current_lr,
            "epoch_duration_sec": round(epoch_duration, 2),
        })

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_checkpoint_path = model_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_name": model_name,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE and epoch > unfreeze_epoch:
                print(f"\n[Early Stopping] Triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60.0:.2f} minutes.")
    print(f"Best Val Accuracy: {best_val_acc:.2f}% (Loss: {best_val_loss:.4f})")

    # Save Final Model Checkpoint
    final_checkpoint_path = model_dir / "final_model.pth"
    torch.save({
        "epoch": len(history),
        "model_state_dict": model.state_dict(),
        "val_loss": history[-1]["val_loss"],
        "val_acc": history[-1]["val_acc"],
        "model_name": model_name,
        "total_training_time_sec": total_training_time,
    }, final_checkpoint_path)

    # Save Training History CSV
    df_history = pd.DataFrame(history)
    history_csv_path = TRAINING_HISTORY_DIR / f"training_history_{model_name}.csv"
    df_history.to_csv(history_csv_path, index=False)

    # Plot & Save Learning Curves
    plot_training_history(df_history, model_name, display_name)

    return df_history, total_training_time

def main():
    parser = argparse.ArgumentParser(description="AgriFresh Model Training Engine")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all' for all 5 models")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (small subset)")
    args = parser.parse_args()

    is_debug = args.debug or DEBUG_MODE
    epochs = args.epochs if args.epochs is not None else (DEBUG_EPOCHS if is_debug else NUM_EPOCHS)

    target_models = MODEL_NAMES if args.model.lower() == "all" else [args.model.lower()]

    print(f"AgriFresh Training Pipeline Initialized | Debug Mode: {is_debug} | Epochs: {epochs}")
    for m in target_models:
        train_model(
            model_name=m,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            debug_mode=is_debug,
        )

    # Automatically evaluate models after full training
    print("\nTraining completed! Running full evaluation across all 5 models...")
    from src.evaluate import evaluate_all_models
    evaluate_all_models(debug_mode=is_debug)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
