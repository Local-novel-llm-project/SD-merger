import os
import json
import time
import logging
import threading
import yaml
from pathlib import Path
from typing import List, Dict, Any

from module.persistence import move_corrupt_file_aside, write_json_file_atomic

HISTORY_FILE = str(Path(__file__).resolve().parent.parent / "merge_history.json")
_HISTORY_LOCK = threading.RLock()


def build_history_metadata(timestamp: float | None = None) -> Dict[str, Any]:
    resolved_timestamp = time.time() if timestamp is None else float(timestamp)
    return {
        "timestamp": resolved_timestamp,
        "date": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(resolved_timestamp)
        ),
    }


def _write_history(history: List[Dict[str, Any]]) -> None:
    write_json_file_atomic(HISTORY_FILE, history, indent=2, ensure_ascii=False)


def load_history() -> List[Dict[str, Any]]:
    with _HISTORY_LOCK:
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                return loaded
            raise ValueError("History file root must be a list.")
        except (json.JSONDecodeError, ValueError) as exc:
            moved_path = None
            try:
                moved_path = move_corrupt_file_aside(HISTORY_FILE)
            except OSError as move_exc:
                logging.error(
                    "History file '%s' was corrupt but could not be moved aside: %s",
                    HISTORY_FILE,
                    move_exc,
                )
            logging.error(
                "History file was corrupt and has been moved aside to '%s': %s",
                moved_path,
                exc,
            )
            return []
        except OSError as exc:
            logging.error("Failed to read history file '%s': %s", HISTORY_FILE, exc)
            return []


def save_history(entry: Dict[str, Any]) -> None:
    with _HISTORY_LOCK:
        history = load_history()
        entry_to_save = dict(entry)
        entry_to_save.update(build_history_metadata())
        history.insert(0, entry_to_save)
        history = history[:100]
        _write_history(history)


def _normalize_output_name(output_name: str) -> tuple[str, str, str, str]:
    normalized = os.path.normcase(os.path.normpath(str(output_name or "").strip()))
    if not normalized:
        return "", "", "", ""

    basename = os.path.basename(normalized)
    stem, suffix = os.path.splitext(basename)
    return normalized, basename, stem, suffix


def _find_matching_history_indexes(
    history: List[Dict[str, Any]],
    output_name: str,
    matcher,
) -> list[int]:
    return [
        index
        for index, entry in enumerate(history)
        if matcher(output_name, entry.get("output_name", ""))
    ]


def _has_directory_component(normalized_path: str, basename: str) -> bool:
    return bool(normalized_path and basename and normalized_path != basename)


def _is_exact_path_output_name_match(candidate: str, recorded: str) -> bool:
    candidate_path, _, _, _ = _normalize_output_name(candidate)
    recorded_path, _, _, _ = _normalize_output_name(recorded)
    if not candidate_path or not recorded_path:
        return False
    return candidate_path == recorded_path


def _is_unique_basename_output_name_match(candidate: str, recorded: str) -> bool:
    candidate_path, candidate_basename, _, candidate_suffix = _normalize_output_name(
        candidate
    )
    recorded_path, recorded_basename, _, recorded_suffix = _normalize_output_name(
        recorded
    )
    if (
        not candidate_basename
        or not recorded_basename
        or candidate_basename != recorded_basename
    ):
        return False

    if candidate_suffix and recorded_suffix and candidate_suffix != recorded_suffix:
        return False

    return _has_directory_component(candidate_path, candidate_basename) or _has_directory_component(
        recorded_path,
        recorded_basename,
    )


def _is_stem_only_output_name_match(candidate: str, recorded: str) -> bool:
    _, _, candidate_stem, candidate_suffix = _normalize_output_name(candidate)
    _, _, recorded_stem, recorded_suffix = _normalize_output_name(recorded)
    if not candidate_stem or not recorded_stem or candidate_stem != recorded_stem:
        return False
    return not candidate_suffix or not recorded_suffix


def _find_history_entry_index(
    history: List[Dict[str, Any]], output_name: str
) -> int | None:
    path_matches = _find_matching_history_indexes(
        history,
        output_name,
        _is_exact_path_output_name_match,
    )
    if path_matches:
        return path_matches[0]

    basename_matches = _find_matching_history_indexes(
        history,
        output_name,
        _is_unique_basename_output_name_match,
    )
    if len(basename_matches) == 1:
        return basename_matches[0]

    stem_matches = [
        index
        for index, entry in enumerate(history)
        if _is_stem_only_output_name_match(output_name, entry.get("output_name", ""))
    ]
    if len(stem_matches) == 1:
        return stem_matches[0]

    return None


def update_history_entry(output_name: str, update_dict: Dict[str, Any]) -> bool:
    """特定の output_name を持つ最新のヒストリエントリを更新する"""
    with _HISTORY_LOCK:
        history = load_history()
        match_index = _find_history_entry_index(history, output_name)
        if match_index is None:
            return False

        history[match_index].update(update_dict)
        _write_history(history)
        return True


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
