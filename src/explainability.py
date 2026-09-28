import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.cm as cm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import get_inference_transform

def disable_inplace_relu(model: torch.nn.Module):
    """
    Ensures all ReLU activations in model have inplace=False to prevent autograd hook issues.
    """
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6)):
            m.inplace = False

def get_target_layer(model: torch.nn.Module, model_name: str):
    """
    Returns the target final convolutional layer for Grad-CAM for each architecture.
    """
    model_key = model_name.lower().replace("-", "_").strip()
    
    if "mobilenetv3" in model_key:
        return model.features[-1]
    elif "efficientnet_b0" in model_key:
        return model.features[-1]
    elif "resnet18" in model_key:
        return model.layer4[-1]
    elif "densenet121" in model_key:
        return model.features
    elif "efficientnet_v2" in model_key:
        return model.features[-1]
    else:
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

def generate_gradcam(model: torch.nn.Module, image_input, model_name: str, target_class_idx: int = None, device: str = None):
    """
    Generates Grad-CAM heatmap overlay for a PIL Image or image path.
    Returns:
    - overlay_image: PIL Image with heatmap overlaid on RGB image.
    - heatmap_pure: PIL Image showing heatmap standalone.
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type. Provide PIL.Image or path.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    disable_inplace_relu(model)
    model.to(dev)
    model.eval()

    # 1. Transform Image
    transform = get_inference_transform()
    tensor_img = transform(pil_img).unsqueeze(0).to(dev)

    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        raise RuntimeError(f"Could not locate target conv layer for Grad-CAM in {model_name}")

    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    h = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(tensor_img)
    probs = F.softmax(output, dim=1)

    if target_class_idx is None:
        target_class_idx = int(torch.argmax(probs, dim=1).item())

    score = output[0, target_class_idx]
    
    # Compute gradients of score w.r.t activations directly
    act_tensor = activations[0]
    grads = torch.autograd.grad(score, act_tensor, retain_graph=True)[0]

    h.remove()

    act = act_tensor.detach().cpu().numpy()[0]   # (C, H, W)
    grad = grads.detach().cpu().numpy()[0]        # (C, H, W)

    # Calculate channel weights via global average pooling of gradients
    weights = np.mean(grad, axis=(1, 2))          # (C,)

    # Weighted sum of feature maps
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    # ReLU to focus on positive contributions
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)

    # Resize CAM to original image size
    orig_w, orig_h = pil_img.size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0

    # Colorize using Matplotlib colormap (compatible with modern matplotlib)
    try:
        colormap = matplotlib.colormaps.get_cmap("jet")
    except AttributeError:
        colormap = cm.get_cmap("jet")

    colored_cam = colormap(cam_arr)[:, :, :3] # RGB channels only [0..1]

    orig_arr = np.array(pil_img) / 255.0
    overlay_arr = 0.5 * orig_arr + 0.5 * colored_cam
    overlay_arr = np.clip(overlay_arr * 255, 0, 255).astype(np.uint8)
    heatmap_pure_arr = np.clip(colored_cam * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay_arr), Image.fromarray(heatmap_pure_arr)
