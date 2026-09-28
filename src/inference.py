import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_NAMES,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
)
from src.preprocessing import get_inference_transform
from src.models import build_model
from src.shelf_life import estimate_produce_shelf_life

# Cache loaded models in memory for fast interactive inference
_LOADED_MODELS = {}

def load_inference_model(model_name: str, device: str = None):
    """
    Loads model weights from checkpoint into cache.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_key = model_name.lower().replace("-", "_").strip()
    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key], device

    model_dir = MODELS_DIR / model_key
    best_checkpoint = model_dir / "best_model.pth"
    final_checkpoint = model_dir / "final_model.pth"

    if not best_checkpoint.exists() and not final_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint for '{model_name}' not found at {model_dir}")

    checkpoint_path = best_checkpoint if best_checkpoint.exists() else final_checkpoint

    model = build_model(model_key, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _LOADED_MODELS[model_key] = model
    return model, device

def predict_single_image(image_input, model_name: str, device: str = None, produce_type: str = None):
    """
    Runs freshness inference on a single image and computes shelf-life estimation.
    image_input: PIL Image or Path or str.
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type. Provide PIL.Image or path.")

    model, dev = load_inference_model(model_name, device=device)
    transform = get_inference_transform()
    tensor_img = transform(image).unsqueeze(0).to(dev)

    import time
    t0 = time.time()
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
    latency_ms = (time.time() - t0) * 1000.0

    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    shelf_info = estimate_produce_shelf_life(predicted_class, prob_dict, produce_type)

    return {
        "model_name": model_name,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100.0, 2),
        "latency_ms": round(latency_ms, 2),
        "probabilities": prob_dict,
        "probabilities_pct": {k: round(v * 100.0, 2) for k, v in prob_dict.items()},
        "shelf_life": shelf_info,
    }

def predict_all_models(image_input, device: str = None):
    """
    Runs the same image across all 5 models and returns comparative predictions.
    """
    results = []
    for m in MODEL_NAMES:
        try:
            res = predict_single_image(image_input, m, device=device)
            results.append({
                "Model": res["display_name"],
                "Prediction": res["predicted_class"],
                "Confidence": f"{res['confidence_pct']}%",
                "Live Latency": f"{res['latency_ms']} ms",
                "Fresh": f"{res['probabilities_pct']['Fresh']}%",
                "Semi-Fresh": f"{res['probabilities_pct']['Semi-Fresh']}%",
                "Rotten": f"{res['probabilities_pct']['Rotten']}%",
                "raw_confidence": res["confidence"],
                "raw_probs": res["probabilities"],
            })
        except Exception as e:
            results.append({
                "Model": MODEL_DISPLAY_NAMES.get(m, m),
                "Prediction": "Error",
                "Confidence": "N/A",
                "Fresh": "N/A",
                "Semi-Fresh": "N/A",
                "Rotten": "N/A",
                "error": str(e),
            })
    return pd.DataFrame(results)
