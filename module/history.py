import os
import json
import time
import yaml
from typing import List, Dict, Any

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "merge_history.json")


def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(entry: Dict[str, Any]) -> None:
    history = load_history()
    entry["timestamp"] = time.time()
    entry["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
    history.insert(0, entry)  # Add to beginning
    # Keep only last 100 entries
    history = history[:100]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def update_history_entry(output_name: str, update_dict: Dict[str, Any]) -> bool:
    """特定の output_name を持つ最新のヒストリエントリを更新する"""
    history = load_history()
    updated = False

    for entry in history:
        if entry.get("output_name") == output_name:
            entry.update(update_dict)
            updated = True
            break

    if updated:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    return updated


def history_to_yaml(entry: dict) -> str:
    """ヒストリエントリからYAML設定を再生成する"""
    config = entry.get("config", {})

    # Optional metadata as comments
    yaml_lines = []
    yaml_lines.append("# Auto-generated merge recipe from history")
    if "date" in entry:
        yaml_lines.append(f"# Date: {entry['date']}")
    if "output_name" in entry:
        yaml_lines.append(f"# Original Output Name: {entry['output_name']}")
    if "status" in entry:
        yaml_lines.append(f"# Status: {entry['status']}")
    yaml_lines.append("")

    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    yaml_lines.append(yaml_content)

    return "\n".join(yaml_lines)


def export_recipe(entry: dict, filepath: str) -> None:
    """ヒストリエントリをYAMLファイルとしてエクスポートする"""
    yaml_str = history_to_yaml(entry)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(yaml_str)
