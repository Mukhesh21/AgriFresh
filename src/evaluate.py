import os
import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_NAMES,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    METRICS_DIR,
    CONFUSION_MATRICES_DIR,
    VISUALIZATIONS_DIR,
    CLASS_NAMES,
    BATCH_SIZE,
)
from src.dataset import get_dataloaders
from src.models import build_model, count_parameters, calculate_model_size_mb
from src.utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve_single,
    plot_roc_curve_comparison,
)

def evaluate_single_model(model_name: str, test_loader, device: torch.device):
    """
    Evaluates a single model on the standardized test set.
    """
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    model_dir = MODELS_DIR / model_name
    best_checkpoint = model_dir / "best_model.pth"
    final_checkpoint = model_dir / "final_model.pth"

    if not best_checkpoint.exists() and not final_checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found for model {model_name} in {model_dir}")

    checkpoint_path = best_checkpoint if best_checkpoint.exists() else final_checkpoint
    print(f"\nEvaluating [{display_name}] using checkpoint: {checkpoint_path.name}")

    # Build model & load weights
    model = build_model(model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    model_size_mb = calculate_model_size_mb(model)

    # Fetch training time from history or checkpoint
    training_time_sec = checkpoint.get("total_training_time_sec", None)
    history_file = RESULTS_DIR / "training_history" / f"training_history_{model_name}.csv"
    if history_file.exists() and training_time_sec is None:
        try:
            df_h = pd.read_csv(history_file)
            if "epoch_duration_sec" in df_h.columns:
                training_time_sec = float(df_h["epoch_duration_sec"].sum())
        except Exception:
            training_time_sec = 0.0
    if training_time_sec is None:
        training_time_sec = 0.0

    y_true = []
    y_pred = []
    y_probs = []
    img_paths = []
    produces = []
    inference_times = []

    with torch.no_grad():
        for images, labels, produce_batch, paths in test_loader:
            images = images.to(device)

            start_t = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_time = (time.perf_counter() - start_t) * 1000.0  # ms
            per_img_time = batch_time / images.size(0)
            inference_times.append(per_img_time)

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)
            img_paths.extend(paths)
            produces.extend(produce_batch)

    avg_inference_time_ms = float(np.mean(inference_times)) if inference_times else 0.0

    metrics = compute_metrics(y_true, y_pred, y_probs)
    metrics["model"] = model_name
    metrics["display_name"] = display_name
    metrics["total_parameters"] = total_params
    metrics["trainable_parameters"] = trainable_params
    metrics["model_size_mb"] = model_size_mb
    metrics["training_time_sec"] = round(training_time_sec, 2)
    metrics["inference_time_ms"] = round(avg_inference_time_ms, 2)

    # Save metrics JSON
    with open(METRICS_DIR / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot confusion matrix & ROC curve
    plot_confusion_matrix(metrics["confusion_matrix"], model_name, display_name)
    plot_roc_curve_single(y_true, y_probs, model_name, display_name)

    print(f"[{display_name}] Results:")
    print(f"  Accuracy:       {metrics['accuracy']*100:.2f}%")
    print(f"  Macro ROC-AUC:  {metrics['roc_auc']*100:.2f}%")
    print(f"  Macro Precision:{metrics['precision']*100:.2f}%")
    print(f"  Macro Recall:   {metrics['recall']*100:.2f}%")
    print(f"  Macro F1:       {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1:    {metrics['weighted_f1']*100:.2f}%")
    print(f"  Params:         {total_params:,}")
    print(f"  Model Size:     {model_size_mb:.2f} MB")
    print(f"  Inference Latency: {avg_inference_time_ms:.2f} ms / image")

    # Prediction dataframe for test instances
    df_preds = pd.DataFrame({
        "file_path": img_paths,
        "produce": produces,
        "true_label_idx": y_true,
        "true_label": [CLASS_NAMES[i] for i in y_true],
        f"pred_label_{model_name}": [CLASS_NAMES[i] for i in y_pred],
        f"confidence_{model_name}": [round(float(np.max(p)), 4) for p in y_probs],
        f"prob_fresh_{model_name}": [round(float(p[0]), 4) for p in y_probs],
        f"prob_semifresh_{model_name}": [round(float(p[1]), 4) for p in y_probs],
        f"prob_rotten_{model_name}": [round(float(p[2]), 4) for p in y_probs],
    })

    return metrics, df_preds, y_true, np.array(y_probs)

def generate_comparative_visualizations(df_comp: pd.DataFrame):
    """
    Generates professional comparative charts across all models.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    models = df_comp["Model"].tolist()
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]

    # 1. Performance Metrics Bar Chart (Accuracy, ROC-AUC, Precision, Recall, F1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    metrics_cols = ["Accuracy", "ROC_AUC", "Precision", "Recall", "F1"]
    avail_cols = [c for c in metrics_cols if c in df_comp.columns]
    x = np.arange(len(models))
    width = 0.15

    for idx, col in enumerate(avail_cols):
        values = df_comp[col].values * 100.0 if df_comp[col].max() <= 1.0 else df_comp[col].values
        ax.bar(x + (idx - 2) * width, values, width, label=col, edgecolor="#2c3e50", alpha=0.9)

    ax.set_title("5-Model Prediction Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "model_performance_comparison.png", dpi=300)
    plt.close()

    # 2. Computational Efficiency: Model Size vs Inference Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(models, df_comp["Model_Size_MB"], color="#34495e", edgecolor="#2c3e50", width=0.55)
    ax1.set_title("Model Disk Size (MB)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Size (MB)")
    ax1.tick_params(axis="x", rotation=25)

    ax2.bar(models, df_comp["Inference_Time"], color="#16a085", edgecolor="#2c3e50", width=0.55)
    ax2.set_title("Avg Inference Latency (ms / image)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Latency (ms)")
    ax2.tick_params(axis="x", rotation=25)

    plt.suptitle("Computational Efficiency Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "model_efficiency_comparison.png", dpi=300)
    plt.close()

    # 3. Accuracy vs Inference Time Trade-off Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    acc_pct = df_comp["Accuracy"].values * 100.0 if df_comp["Accuracy"].max() <= 1.0 else df_comp["Accuracy"].values
    scatter = ax.scatter(
        df_comp["Inference_Time"],
        acc_pct,
        s=df_comp["Model_Size_MB"] * 8 + 100,
        c=range(len(models)),
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=1.5,
    )
    for i, txt in enumerate(models):
        ax.annotate(
            f" {txt}\n ({df_comp['Model_Size_MB'].iloc[i]} MB)",
            (df_comp["Inference_Time"].iloc[i], acc_pct[i]),
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Accuracy vs. Inference Latency (Bubble size = Model Size)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Average Inference Time (ms)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "accuracy_vs_latency.png", dpi=300)
    plt.close()

    print(f"Comparative visualizations saved to: {VISUALIZATIONS_DIR}")

def evaluate_all_models(debug_mode: bool = False):
    """
    Evaluates all 5 models and compiles model_comparison.csv and predictions.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("EVALUATING ALL 5 TRAINED MODELS ON STANDARDIZED TEST SET")
    print("=" * 70)

    _, _, test_loader, _, _, df_test = get_dataloaders(debug_mode=debug_mode)

    all_metrics = []
    combined_preds = None
    roc_data_dict = {}

    for m in MODEL_NAMES:
        try:
            m_metrics, m_preds, y_t, y_p = evaluate_single_model(m, test_loader, device)
            display_name = MODEL_DISPLAY_NAMES.get(m, m)
            all_metrics.append({
                "Model": display_name,
                "model_key": m,
                "Accuracy": round(m_metrics["accuracy"], 4),
                "ROC_AUC": round(m_metrics["roc_auc"], 4),
                "Precision": round(m_metrics["precision"], 4),
                "Recall": round(m_metrics["recall"], 4),
                "F1": round(m_metrics["f1"], 4),
                "Macro_F1": round(m_metrics["macro_f1"], 4),
                "Weighted_F1": round(m_metrics["weighted_f1"], 4),
                "Parameters": m_metrics["total_parameters"],
                "Trainable_Parameters": m_metrics["trainable_parameters"],
                "Model_Size_MB": m_metrics["model_size_mb"],
                "Training_Time": round(m_metrics["training_time_sec"], 2),
                "Inference_Time": round(m_metrics["inference_time_ms"], 2),
            })
            roc_data_dict[m] = {
                "display_name": display_name,
                "y_true": y_t,
                "y_probs": y_p,
            }

            if combined_preds is None:
                combined_preds = m_preds
            else:
                pred_cols = [c for c in m_preds.columns if m in c]
                combined_preds = pd.concat([combined_preds, m_preds[pred_cols]], axis=1)
        except Exception as e:
            print(f"Error evaluating model {m}: {e}")

    if not all_metrics:
        print("No models were successfully evaluated.")
        return None

    # Plot combined ROC-AUC comparison curve across all models
    plot_roc_curve_comparison(roc_data_dict)

    df_comp = pd.DataFrame(all_metrics)
    comparison_csv = RESULTS_DIR / "model_comparison.csv"
    df_comp.to_csv(comparison_csv, index=False)

    if combined_preds is not None:
        predictions_csv = RESULTS_DIR / "predictions.csv"
        combined_preds.to_csv(predictions_csv, index=False)

    print("\n" + "=" * 70)
    print("FINAL 5-MODEL COMPARATIVE BENCHMARK (WITH ROC-AUC)")
    print("=" * 70)
    print(df_comp.to_string(index=False))
    print("=" * 70)

    generate_comparative_visualizations(df_comp)
    return df_comp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AgriFresh Models")
    parser.add_argument("--debug", action="store_true", help="Evaluate on debug test subset")
    args = parser.parse_args()
    evaluate_all_models(debug_mode=args.debug)
