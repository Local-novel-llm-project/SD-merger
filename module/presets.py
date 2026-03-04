import os
import json
from typing import Dict, Any, List

PRESET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "presets"))


def init_presets(category: str = "default") -> str:
    cat_dir = os.path.join(PRESET_DIR, category)
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)
    return cat_dir


def list_presets(category: str = "default") -> List[str]:
    cat_dir = init_presets(category)
    files = [f for f in os.listdir(cat_dir) if f.endswith(".json")]
    return sorted(files)


def load_preset(name: str, category: str = "default") -> Dict[str, Any]:
    cat_dir = init_presets(category)
    path = os.path.join(cat_dir, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_preset(name: str, data: Dict[str, Any], category: str = "default") -> str:
    cat_dir = init_presets(category)
    if not name.endswith(".json"):
        name += ".json"
    path = os.path.join(cat_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return f"Preset saved to {path}"
