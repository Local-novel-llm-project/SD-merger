import os
import json
import gradio as gr

PRESET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "presets")
)


def init_presets():
    if not os.path.exists(PRESET_DIR):
        os.makedirs(PRESET_DIR)


def list_presets():
    init_presets()
    files = [f for f in os.listdir(PRESET_DIR) if f.endswith(".json")]
    return sorted(files)


def load_preset(name):
    path = os.path.join(PRESET_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_preset(name, data):
    init_presets()
    if not name.endswith(".json"):
        name += ".json"
    path = os.path.join(PRESET_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return f"Preset saved to {path}"
