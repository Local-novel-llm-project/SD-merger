import os
import json
import time

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "merge_history.json")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(entry):
    history = load_history()
    entry["timestamp"] = time.time()
    entry["date"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"])
    )
    history.insert(0, entry)  # Add to beginning
    # Keep only last 100 entries
    history = history[:100]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
