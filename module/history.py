import os
import json
import time
import yaml

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
    entry["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
    history.insert(0, entry)  # Add to beginning
    # Keep only last 100 entries
    history = history[:100]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


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
