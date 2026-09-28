import sys
import re
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cache high-accuracy OOD gatekeeper model in memory
_OOD_MODEL = None
_OOD_WEIGHTS = None

PRODUCE_KEYWORDS = {
    "banana", "orange", "lemon", "lime", "pineapple", "strawberry", "fig",
    "pomegranate", "custard apple", "jackfruit", "cucumber", "zucchini",
    "squash", "spaghetti squash", "acorn squash", "bell pepper", "cabbage",
    "head cabbage", "broccoli", "mushroom", "granny smith", "apple", "mango",
    "papaya", "eggplant", "aubergine", "tomato", "cardoon", "artichoke",
    "plum", "peach", "apricot", "cherry", "grape", "gourd", "potato", "onion",
    "corn", "ear", "produce", "fruit", "vegetable", "kiwi", "avocado",
    "guava", "watermelon", "melon", "carrot", "garlic", "ginger",
    "beetroot", "radish", "turnip", "pumpkin", "lettuce", "spinach",
}

NON_PRODUCE_KEYWORDS = {
    # Confectionery & sweets — the most common OOD failure class
    # ImageNet calls chocolates 'bonbon', 'truffle', 'praline' etc.
    "confectionery", "chocolate", "chocolate sauce", "bonbon", "truffle",
    "praline", "nougat", "fudge", "toffee", "caramel", "candy bar",
    "candy", "sweet", "lollipop", "gummy", "marshmallow", "licorice",
    "peppermint", "jawbreaker", "cotton candy", "marzipan", "fondant",
    "truffles", "ganache", "brittle",
    # Baked goods
    "bakery", "dough", "waffle", "pretzel", "bagel", "cookie", "cake",
    "muffin", "croissant", "biscuit", "brownie", "cupcake", "donut",
    "bread", "roll", "loaf", "pastry", "tart", "pie",
    # Prepared / processed foods
    "pizza", "trifle", "plate", "dish", "menu", "espresso", "ice cream",
    "hotdog", "cheeseburger", "hamburger", "burrito", "sushi", "nachos",
    "french fries", "fried chicken", "pasta", "noodle", "soup",
    "sandwich", "wrap", "sauce", "ketchup", "mustard",
    # Non-food objects
    "envelope", "web site", "paper", "book", "binder", "comic book", "notebook",
    "book jacket", "book cover", "magazine", "booklet", "pamphlet", "catalog",
    "poster", "picture frame", "illustration", "artwork", "painting",
    "vehicle", "car", "truck", "bicycle", "furniture", "clothing",
    "curtain", "screen", "monitor", "packet", "carton",
    "dining table", "restaurant", "pillow", "teddy", "cup", "coffee mug",
    "toy", "card", "remote control", "keyboard", "phone",
    "wallet", "pen", "watch", "sunglasses", "hat", "shoe",
    "box", "package", "container", "tin", "jar", "bottle", "can", "hand", "finger",
}

def get_ood_model():
    global _OOD_MODEL, _OOD_WEIGHTS
    if _OOD_MODEL is None:
        _OOD_WEIGHTS = models.ResNet50_Weights.DEFAULT
        _OOD_MODEL = models.resnet50(weights=_OOD_WEIGHTS)
        _OOD_MODEL.eval()
    return _OOD_MODEL, _OOD_WEIGHTS

def is_word_match(keyword: str, text: str) -> bool:
    """Matches keyword as a whole word or clean phrase."""
    pattern = r'\b' + re.escape(keyword) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

def verify_produce_image(image_input) -> tuple:
    """
    High-accuracy zero-shot produce verification using ResNet50.
    Compares cumulative produce vs non-produce probability mass.
    Returns:
    - is_produce: bool
    - confidence_score: float (0.0 to 1.0)
    - detected_label: str (most likely category name)
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input. Pass PIL.Image or path.")

    model, weights = get_ood_model()
    preprocess = weights.transforms()
    tensor_img = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor_img)
        probs = F.softmax(output, dim=1)[0]

    # Cast a wider net: check top-10 predictions (was 5)
    top_k = 10
    top_prob, top_catid = torch.topk(probs, top_k)

    categories = weights.meta["categories"]
    top_label = categories[top_catid[0].item()]
    top_conf = float(top_prob[0].item())

    produce_score = 0.0
    non_produce_score = 0.0
    first_produce_label = None
    first_non_produce_label = None

    for prob, catid in zip(top_prob, top_catid):
        cat_name = categories[catid.item()].lower()
        p_val = float(prob.item())

        is_prod = any(is_word_match(kw, cat_name) for kw in PRODUCE_KEYWORDS)
        is_non_prod = any(is_word_match(kw, cat_name) for kw in NON_PRODUCE_KEYWORDS)

        if is_prod:
            produce_score += p_val
            if first_produce_label is None:
                first_produce_label = categories[catid.item()]

        if is_non_prod:
            non_produce_score += p_val
            if first_non_produce_label is None:
                first_non_produce_label = categories[catid.item()]

    top_is_produce = any(is_word_match(kw, top_label.lower()) for kw in PRODUCE_KEYWORDS)
    top_is_non_produce = any(is_word_match(kw, top_label.lower()) for kw in NON_PRODUCE_KEYWORDS)

    # Rule 1: Strong produce signal — must beat non_produce by a clear margin (>=0.05)
    if (produce_score >= non_produce_score + 0.05) and (produce_score >= 0.15 or top_is_produce):
        return True, max(produce_score, top_conf), first_produce_label or top_label

    # Rule 2: Explicit non-produce signal — lowered threshold from 0.15 → 0.10
    if non_produce_score >= 0.10 or top_is_non_produce:
        return False, max(non_produce_score, top_conf), first_non_produce_label or top_label

    # Rule 3: High-confidence prediction in unrelated class — lowered from 0.40 → 0.35
    if top_conf >= 0.35 and produce_score < 0.05:
        return False, top_conf, top_label

    # Rule 4: Neither keyword matched at all — SAFE DEFAULT: reject ambiguous inputs
    # Previously this returned True (passed through). Now we block unknown inputs.
    if produce_score < 0.05 and non_produce_score < 0.05:
        return False, top_conf, f"Unrecognized Input ({top_label})"

    # Rule 5: Marginal produce evidence — allow through
    return True, top_conf, f"Agricultural Specimen ({top_label})"
