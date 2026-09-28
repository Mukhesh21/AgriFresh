import os
import sys
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, CONFUSION_MATRICES_DIR, VISUALIZATIONS_DIR

ROC_DIR = VISUALIZATIONS_DIR / "roc_curves"
ROC_DIR.mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Computes comprehensive multi-class evaluation metrics including ROC-AUC.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    roc_auc_macro = 0.0
    roc_auc_per_class = {}

    if y_probs is not None:
        y_probs = np.array(y_probs)
        try:
            roc_auc_macro = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            for idx, name in enumerate(CLASS_NAMES):
                if y_true_bin.shape[1] > idx and y_probs.shape[1] > idx:
                    roc_auc_per_class[name] = float(roc_auc_score(y_true_bin[:, idx], y_probs[:, idx]))
                else:
                    roc_auc_per_class[name] = 0.0
        except Exception as e:
            print(f"Warning computing ROC-AUC: {e}")

    per_class_dict = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_dict[name] = {
            "precision": float(prec_per_class[idx]) if idx < len(prec_per_class) else 0.0,
            "recall": float(rec_per_class[idx]) if idx < len(rec_per_class) else 0.0,
            "f1": float(f1_per_class[idx]) if idx < len(f1_per_class) else 0.0,
            "roc_auc": float(roc_auc_per_class.get(name, 0.0)),
        }

    return {
        "accuracy": float(acc),
        "precision": float(prec_macro),
        "recall": float(rec_macro),
        "f1": float(f1_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "roc_auc": float(roc_auc_macro),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_dict,
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        ),
    }

def plot_confusion_matrix(cm_array, model_name: str, display_name: str, save_path: Path = None):
    """
    Plots and saves a professional confusion matrix heatmap.
    """
    cm = np.array(cm_array)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        linewidths=1.0,
        linecolor="#bdc3c7",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {display_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = CONFUSION_MATRICES_DIR / f"cm_{model_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def plot_training_history(history_df: pd.DataFrame, model_name: str, display_name: str, save_path: Path = None):
    """
    Plots training vs validation accuracy and loss curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(history_df["epoch"], history_df["train_loss"], "o-", label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(history_df["epoch"], history_df["val_loss"], "s-", label="Val Loss", color="#3498db", linewidth=2)
    ax1.set_title(f"{display_name} - Loss History", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy curve
    ax2.plot(history_df["epoch"], history_df["train_acc"], "o-", label="Train Acc", color="#27ae60", linewidth=2)
    ax2.plot(history_df["epoch"], history_df["val_acc"], "s-", label="Val Acc", color="#8e44ad", linewidth=2)
    ax2.set_title(f"{display_name} - Accuracy History", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = VISUALIZATIONS_DIR / f"history_{model_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def plot_roc_curve_single(y_true, y_probs, model_name: str, display_name: str, save_path: Path = None):
    """
    Plots and saves multi-class One-vs-Rest ROC curve for a single model.
    """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    y_probs = np.array(y_probs)

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(7, 5.5))

    macro_roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")

    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"ROC {class_name} (AUC = {roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold")
    ax.set_title(f"ROC-AUC Curve: {display_name} (Macro AUC = {macro_roc_auc:.4f})", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path is None:
        save_path = ROC_DIR / f"roc_{model_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path, macro_roc_auc

def plot_roc_curve_comparison(all_models_data: dict, save_path: Path = None):
    """
    Plots a combined ROC comparison curve for all models on one graph (Micro/Macro Average).
    `all_models_data` structure: { model_name: { 'display_name': str, 'y_true': list, 'y_probs': ndarray } }
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (model_name, data), color in zip(all_models_data.items(), palette):
        y_true = np.array(data["y_true"])
        y_probs = np.array(data["y_probs"])
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        
        # Compute micro-average ROC curve and ROC area
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        display_name = data.get("display_name", model_name)
        ax.plot(fpr_micro, tpr_micro, color=color, lw=2, label=f"{display_name} (AUC = {roc_auc_micro:.4f})")

    ax.plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title("Comparative Micro-Averaged ROC Curves (All Models)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path is None:
        save_path = ROC_DIR / "roc_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
