"""
AgriFresh - Real-Time Shelf-Life & Freshness Index Estimation Engine
Calculates remaining shelf-life days, continuous Freshness Index Score (0-100%),
and produce-specific storage/preservation recommendations.
"""

from typing import Dict, Any, Optional

# Domain-expert agricultural shelf-life database derived from post-harvest research & AgriFreshNET annotations
PRODUCE_SHELF_LIFE_DATABASE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Banana": {
        "Fresh": {
            "days_range": "1 - 4 Days Remaining",
            "status": "Optimal Freshness (Unblemished Skin)",
            "storage_tip": "Store at 13-15C (room temperature). Keep away from direct sunlight. Do not refrigerate unpeeled bananas to prevent chilling injury.",
            "action": "Grade A: Ideal for retail, export, or supermarket display."
        },
        "Semi-Fresh": {
            "days_range": "1 - 3 Days Remaining",
            "status": "Maturing / Ripening Stage",
            "storage_tip": "Consume immediately or refrigerate to slow further ethylene release. Great for smoothies, baking, or processing.",
            "action": "Grade B: Prioritize for immediate sale or food processing."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Necrotic Breakdown & High Ethylene",
            "storage_tip": "Isolate immediately. High ethylene emission accelerates ripening and decay in neighboring produce.",
            "action": "Unfit for consumption: Divert to composting or biogas production."
        }
    },
    "Bittermelon": {
        "Fresh": {
            "days_range": "1 - 3 Days Remaining",
            "status": "Optimal Crispness & Turgor",
            "storage_tip": "Store in crisping drawer at 10-12C with 85-90% relative humidity. Keep dry.",
            "action": "Grade A: Premium fresh market quality."
        },
        "Semi-Fresh": {
            "days_range": "1 - 2 Days Remaining",
            "status": "Softening & Early Yellowing",
            "storage_tip": "Wrap in perforated plastic and refrigerate. Use promptly for cooking.",
            "action": "Grade B: Consume or cook within 48 hours."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Soft Rot & Fungal Spores",
            "storage_tip": "Discard. Soft rot bacterial breakdown creates contamination risks.",
            "action": "Unfit: Reject batch."
        }
    },
    "Cucumber": {
        "Fresh": {
            "days_range": "1 - 6 Days Remaining",
            "status": "Optimal Firmness & Hydration",
            "storage_tip": "Store at 10-12C. Keep dry and avoid ethylene exposure (keep away from bananas and tomatoes).",
            "action": "Grade A: Commercial long-haul stability."
        },
        "Semi-Fresh": {
            "days_range": "2 - 4 Days Remaining",
            "status": "Minor Moisture Loss / Softening",
            "storage_tip": "Wrap in food wrap and refrigerate in crisper drawer.",
            "action": "Grade B: Suitable for local sales; avoid long transport."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Watery Breakdown & Pitting",
            "storage_tip": "Discard immediately to prevent bacterial soft-rot infection in adjacent stock.",
            "action": "Unfit: Commercial rejection."
        }
    },
    "Eggplant": {
        "Fresh": {
            "days_range": "1 - 4 Days Remaining",
            "status": "High Skin Gloss & Firmness",
            "storage_tip": "Store at 10-12C with 90% humidity. Avoid temperatures below 7C to prevent chilling injury.",
            "action": "Grade A: Premium retail condition."
        },
        "Semi-Fresh": {
            "days_range": "1 - 3 Days Remaining",
            "status": "Loss of Gloss & Sub-surface Softness",
            "storage_tip": "Keep in cool storage. Cook within 48-72 hours to prevent internal seed browning.",
            "action": "Grade B: Recommend quick culinary sale."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Spongy Necrosis & Internal Decay",
            "storage_tip": "Discard. Internal fungal breakdown present.",
            "action": "Unfit: Unsafe for human consumption."
        }
    },
    "Orange": {
        "Fresh": {
            "days_range": "1 - 9 Days (Ambient) / Up to 3 Weeks (Refrigerated)",
            "status": "Optimal Juice Content & Rind Firmness",
            "storage_tip": "Store at 4-7C for extended shelf life, or in cool ambient ventilated storage for up to 9 days.",
            "action": "Grade A: High export stability."
        },
        "Semi-Fresh": {
            "days_range": "3 - 6 Days Remaining",
            "status": "Rind Dehydration",
            "storage_tip": "Refrigerate promptly or process into fresh juice.",
            "action": "Grade B: Ideal for juicing and beverage extraction."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Blue/Green Penicillium Mold",
            "storage_tip": "Remove and segregate immediately. Mold spores propagate rapidly through air.",
            "action": "Unfit: Dispose of safely."
        }
    },
    "Papaya": {
        "Fresh": {
            "days_range": "1 - 4 Days Remaining",
            "status": "Firm Ripe Condition",
            "storage_tip": "Store at room temperature until fully golden, then refrigerate at 7-10C.",
            "action": "Grade A: Standard commercial distribution."
        },
        "Semi-Fresh": {
            "days_range": "1 - 2 Days Remaining",
            "status": "Fully Ripe / Softening Tissue",
            "storage_tip": "Refrigerate and consume within 24-48 hours.",
            "action": "Grade B: Immediate consumption or fruit puree."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Sunken Anthracnose Lesions",
            "storage_tip": "Discard immediately.",
            "action": "Unfit: Reject."
        }
    },
    "Pineapple": {
        "Fresh": {
            "days_range": "1 - 15 Days Remaining",
            "status": "Optimal Sugar-Acid Equilibrium",
            "storage_tip": "Store at 7-10C. Note: Pineapples do not continue to ripen post-harvest.",
            "action": "Grade A: Extended fresh market life."
        },
        "Semi-Fresh": {
            "days_range": "3 - 7 Days Remaining",
            "status": "Aroma Intensification & Base Softening",
            "storage_tip": "Remove crown, cut into chunks, and store refrigerated in an airtight container.",
            "action": "Grade B: Divert to fresh-cut fruit cups or catering."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Fermentation & Basal Decay",
            "storage_tip": "Discard immediately.",
            "action": "Unfit: Reject shipment."
        }
    },
    "Tomato": {
        "Fresh": {
            "days_range": "1 - 10 Days Remaining",
            "status": "Optimal Firmness & Flavor Profile",
            "storage_tip": "Store stem-side down at 15-20C. Avoid refrigerating below 12C to preserve aromatic flavor volatiles.",
            "action": "Grade A: Prime fresh market produce."
        },
        "Semi-Fresh": {
            "days_range": "2 - 5 Days Remaining",
            "status": "Deep Red / Softening Rind",
            "storage_tip": "Ideal for cooking, sauces, soups, or tomato paste.",
            "action": "Grade B: Commercial sauce and paste processing."
        },
        "Rotten": {
            "days_range": "0 Days (Expired / Spoiled)",
            "status": "Fungal Rot & Skin Rupture",
            "storage_tip": "Discard immediately to prevent fruit fly attractance and mold spread.",
            "action": "Unfit: Reject."
        }
    }
}

DEFAULT_SHELF_LIFE_GENERIC = {
    "Fresh": {
        "days_range": "1 - 5 Days Remaining",
        "status": "Optimal Freshness",
        "storage_tip": "Store in cool, dry conditions with adequate ventilation.",
        "action": "Grade A: Safe for sale and consumption."
    },
    "Semi-Fresh": {
        "days_range": "1 - 3 Days Remaining",
        "status": "Maturing / Consume Soon",
        "storage_tip": "Refrigerate and consume promptly.",
        "action": "Grade B: Fast-track for consumption or processing."
    },
    "Rotten": {
        "days_range": "0 Days (Expired / Spoiled)",
        "status": "Spoiled / Unfit",
        "storage_tip": "Dispose of safely to avoid cross-contamination.",
        "action": "Unfit for consumption."
    }
}

CROP_SYNONYMS: Dict[str, list] = {
    "Banana": ["banana", "plantain"],
    "Bittermelon": ["bittermelon", "bitter", "gourd", "karela"],
    "Cucumber": ["cucumber", "cuke", "zucchini", "squash", "gherkin"],
    "Eggplant": ["eggplant", "aubergine", "brinjal"],
    "Orange": ["orange", "lemon", "lime", "citrus", "mandarin", "grapefruit", "tangerine"],
    "Papaya": ["papaya", "pawpaw", "papaw"],
    "Pineapple": ["pineapple", "ananas"],
    "Tomato": ["tomato"]
}

def resolve_crop_name(
    selected_crop: Optional[str] = None,
    detected_label: Optional[str] = None,
    filename: Optional[str] = None
) -> str:
    """
    Intelligently maps input, filename, or OOD keywords to one of the 8 supported project crops.
    Avoids leaking arbitrary ImageNet labels like 'fig' or 'cardoon'.
    """
    # 1. User manual selection has highest priority
    if selected_crop and selected_crop not in ["Auto-Detect Crop", "General", "None", ""]:
        for p_name in PRODUCE_SHELF_LIFE_DATABASE.keys():
            if p_name.lower() == selected_crop.lower():
                return p_name

    # 2. Check filename clues
    if filename:
        fn_lower = filename.lower()
        for primary_crop, synonyms in CROP_SYNONYMS.items():
            if any(syn in fn_lower for syn in synonyms):
                return primary_crop

    # 3. Check OOD Gatekeeper detected label against project crop taxonomy
    if detected_label:
        dl_lower = detected_label.lower()
        for primary_crop, synonyms in CROP_SYNONYMS.items():
            if any(syn in dl_lower for syn in synonyms):
                return primary_crop

    # 4. Fallback for unclassified agricultural produce
    return "Produce (Auto-Detected)"

def calculate_freshness_index(probabilities: Dict[str, float]) -> float:
    """
    Computes a continuous Freshness Index Score (0.0% to 100.0%) from multi-class probabilities.
    Formula: (P_Fresh * 1.0 + P_SemiFresh * 0.5 + P_Rotten * 0.0) * 100
    """
    p_fresh = probabilities.get("Fresh", 0.0)
    p_semi = probabilities.get("Semi-Fresh", 0.0)
    
    # If probabilities are given as percentages (0-100), convert to 0-1
    if p_fresh > 1.0 or p_semi > 1.0:
        p_fresh /= 100.0
        p_semi /= 100.0

    index = (p_fresh * 1.0 + p_semi * 0.5) * 100.0
    return round(float(index), 1)

def estimate_produce_shelf_life(
    predicted_class: str,
    probabilities: Dict[str, float],
    produce_type: Optional[str] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculates remaining shelf life, freshness index, and tailored storage guidelines.
    """
    crop_label = resolve_crop_name(produce_type, produce_type, filename)

    if crop_label in PRODUCE_SHELF_LIFE_DATABASE:
        crop_data = PRODUCE_SHELF_LIFE_DATABASE[crop_label]
    else:
        crop_data = DEFAULT_SHELF_LIFE_GENERIC

    class_info = crop_data.get(predicted_class, DEFAULT_SHELF_LIFE_GENERIC.get(predicted_class, DEFAULT_SHELF_LIFE_GENERIC["Fresh"]))
    freshness_index = calculate_freshness_index(probabilities)

    return {
        "produce_crop": crop_label,
        "predicted_class": predicted_class,
        "freshness_index": freshness_index,
        "estimated_shelf_life": class_info["days_range"],
        "status_description": class_info["status"],
        "storage_recommendation": class_info["storage_tip"],
        "recommended_action": class_info["action"],
    }
