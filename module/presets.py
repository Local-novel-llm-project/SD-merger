import os
import json
from pathlib import Path
from typing import Dict, Any, List

PRESET_DIR = str(Path(__file__).resolve().parent.parent / "presets")


def _ensure_relative_name(value: str, *, label: str, default: str | None = None) -> str:
    normalized = os.path.normpath(str(value or default or "").strip())
    if normalized in ("", "."):
        if default is None:
            raise ValueError(f"{label} is required.")
        normalized = default

    separators = {os.sep}
    if os.altsep:
        separators.add(os.altsep)

    if os.path.isabs(normalized) or normalized.startswith(".."):
        raise ValueError(f"{label} must stay within the presets directory.")
    if any(separator in normalized for separator in separators):
        raise ValueError(f"{label} must not contain directory separators.")

    return normalized


def _ensure_within_root(root_dir: str, path: str) -> str:
    resolved_root = os.path.abspath(root_dir)
    resolved_path = os.path.abspath(path)
    if resolved_path == resolved_root or resolved_path.startswith(resolved_root + os.sep):
        return resolved_path
    raise ValueError("Preset path must stay within the presets directory.")


def _resolve_category_dir(category: str = "default") -> str:
    category_name = _ensure_relative_name(
        category,
        label="Preset category",
        default="default",
    )
    return _ensure_within_root(PRESET_DIR, os.path.join(PRESET_DIR, category_name))


def _resolve_preset_path(name: str, category: str = "default") -> str:
    category_dir = _resolve_category_dir(category)
    preset_name = _ensure_relative_name(name, label="Preset name")
    if not preset_name.endswith(".json"):
        preset_name += ".json"
    return _ensure_within_root(category_dir, os.path.join(category_dir, preset_name))


def init_presets(category: str = "default") -> str:
    cat_dir = _resolve_category_dir(category)
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)
    return cat_dir


def list_presets(category: str = "default") -> List[str]:
    cat_dir = init_presets(category)
    files = [f for f in os.listdir(cat_dir) if f.endswith(".json")]
    return sorted(files)


def load_preset(name: str, category: str = "default") -> Dict[str, Any]:
    path = _resolve_preset_path(name, category)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_preset(name: str, data: Dict[str, Any], category: str = "default") -> str:
    init_presets(category)
    path = _resolve_preset_path(name, category)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return f"Preset saved to {path}"
