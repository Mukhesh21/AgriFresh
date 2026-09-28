"""
AgriFresh - High-Throughput Batch Produce Inspection Engine
Processes multiple image specimens concurrently, performing:
1. Zero-shot OOD Gatekeeping
2. Neural Freshness Classification
3. Shelf-Life and Freshness Index scoring
4. Generates structured inspection summary DataFrame & CSV report.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
from PIL import Image
import pandas as pd
import time

from src.inference import predict_single_image, load_inference_model
from src.shelf_life import estimate_produce_shelf_life, resolve_crop_name
from src.ood_gatekeeper import verify_produce_image

def process_batch_images(
    images_list: List[Union[str, Path, Image.Image]],
    image_names: List[str],
    model_name: str = "densenet121",
    enable_ood: bool = True,
    selected_crop: str = "Auto-Detect Crop",
    device: str = None
) -> pd.DataFrame:
    """
    Processes a list of produce images and returns a consolidated summary DataFrame.
    """
    records = []
    
    for idx, (img_item, img_name) in enumerate(zip(images_list, image_names)):
        t_start = time.time()
        
        # Load image
        if isinstance(img_item, (str, Path)):
            pil_img = Image.open(img_item).convert("RGB")
        elif isinstance(img_item, Image.Image):
            pil_img = img_item.convert("RGB")
        else:
            records.append({
                "Batch ID": f"LOT-{idx+1:03d}",
                "Filename": img_name,
                "Status": "Error",
                "Produce Crop": "Unknown",
                "Freshness State": "Read Error",
                "Freshness Index (%)": 0.0,
                "Confidence (%)": 0.0,
                "Estimated Shelf-Life": "N/A",
                "Quality Action": "Invalid Image Format",
                "Latency (ms)": 0.0
            })
            continue

        # 1. OOD Verification
        detected_crop = None
        if enable_ood:
            is_valid, ood_conf, detected_label = verify_produce_image(pil_img)
            if not is_valid:
                records.append({
                    "Batch ID": f"LOT-{idx+1:03d}",
                    "Filename": img_name,
                    "Status": "Intercepted (OOD)",
                    "Produce Crop": detected_label,
                    "Freshness State": "Non-Produce Rejection",
                    "Freshness Index (%)": 0.0,
                    "Confidence (%)": round(ood_conf * 100.0, 1),
                    "Estimated Shelf-Life": "0 Days (Non-Food/OOD)",
                    "Quality Action": "Rejected by Safety Filter",
                    "Latency (ms)": round((time.time() - t_start) * 1000.0, 2)
                })
                continue
            else:
                detected_crop = detected_label

        # 2. Freshness Prediction with intelligent crop taxonomy resolution
        resolved_crop = resolve_crop_name(selected_crop, detected_crop, img_name)
        res = predict_single_image(pil_img, model_name, device=device, produce_type=resolved_crop)
        
        shelf_info = res.get("shelf_life", {})
        
        records.append({
            "Batch ID": f"LOT-{idx+1:03d}",
            "Filename": img_name,
            "Status": "Passed",
            "Produce Crop": resolved_crop,
            "Freshness State": res["predicted_class"],
            "Freshness Index (%)": shelf_info.get("freshness_index", 0.0),
            "Confidence (%)": res["confidence_pct"],
            "Estimated Shelf-Life": shelf_info.get("estimated_shelf_life", "N/A"),
            "Quality Action": shelf_info.get("recommended_action", "N/A"),
            "Latency (ms)": res["latency_ms"]
        })

    return pd.DataFrame(records)
