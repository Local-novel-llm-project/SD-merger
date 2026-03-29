from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from module.config_schema import MergeConfig
from module.history import export_recipe, history_to_yaml, load_history
from ui.utils import enqueue_merge_task


def _paths_match(left: object, right: object) -> bool:
    if left in (None, "") or right in (None, ""):
        return False

    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def _resolve_history_left_right_velocity(
    config: dict,
    primary_model: dict,
) -> str:
    for key in ("left_right_velocity", "lrv", "lr", "strategy_velocity"):
        value = primary_model.get(key)
        if value not in (None, ""):
            return str(value)

    velocity = primary_model.get("velocity", "")
    if velocity in (None, ""):
        return ""

    target_model = config.get("target_model")
    if target_model in (None, "") or _paths_match(target_model, primary_model.get("left")):
        return str(velocity)

    return "1.0"


def build_history_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for entry in load_history():
        config = entry.get("config") or {}
        models = config.get("models") or []
        primary_model = models[0] if models else {}

        if primary_model:
            strategy = str(primary_model.get("strategy") or "")
            model_a = str(primary_model.get("left") or "")
            model_b = str(primary_model.get("right") or "")
            velocity = primary_model.get("velocity", "")
            left_right_velocity = _resolve_history_left_right_velocity(
                config,
                primary_model,
            )
        else:
            strategy = "target-only"
            model_a = str(config.get("target_model") or "")
            model_b = ""
            velocity = ""
            left_right_velocity = ""

        rows.append(
            {
                "date": str(entry.get("date") or ""),
                "output_name": str(entry.get("output_name") or ""),
                "status": str(entry.get("status") or ""),
                "strategy": strategy,
                "model_a": model_a,
                "model_b": model_b,
                "velocity": "" if velocity == "" else str(velocity),
                "left_right_velocity": (
                    "" if left_right_velocity == "" else str(left_right_velocity)
                ),
            }
        )

    return rows


def resolve_output_name(config: dict[str, Any], fallback: str = "imported_recipe") -> str:
    output_name = str(config.get("output_name") or "").strip()
    if output_name:
        return output_name

    for model_config in config.get("models") or []:
        if not isinstance(model_config, dict):
            continue
        nested_output_name = str(model_config.get("output_name") or "").strip()
        if nested_output_name:
            return nested_output_name

    return fallback


def find_history_entry(output_name: str) -> dict | None:
    for entry in load_history():
        if str(entry.get("output_name") or "") == output_name:
            return entry
    return None


def get_history_entry(output_name: str) -> dict:
    entry = find_history_entry(output_name)
    if entry is None:
        raise ValueError(f"History entry was not found: {output_name}")
    return entry


def build_history_yaml(output_name: str) -> str:
    return history_to_yaml(get_history_entry(output_name))


def parse_history_yaml_text(yaml_text: str) -> dict[str, Any]:
    if not yaml_text.strip():
        raise ValueError("YAML is empty.")

    try:
        loaded = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError("YAML root must be a mapping.")

    try:
        return MergeConfig(**loaded).model_dump()
    except ValidationError as exc:
        raise ValueError(f"Invalid merge configuration: {exc}") from exc


def queue_history_yaml(
    yaml_text: str,
    fallback_output_name: str = "imported_recipe",
) -> tuple[str, str]:
    config = parse_history_yaml_text(yaml_text)
    output_name = resolve_output_name(config, fallback=fallback_output_name)
    task_id = enqueue_merge_task(
        config,
        output_name,
        task_name=f"Rerun: {output_name}",
    )
    return task_id, output_name


def load_yaml_from_path(path_text: str) -> str:
    path = Path(path_text.strip())
    if not path.name:
        raise ValueError("YAML path is required.")
    if not path.exists():
        raise ValueError(f"YAML file was not found: {path}")
    if not path.is_file():
        raise ValueError(f"YAML path is not a file: {path}")
    return path.read_text(encoding="utf-8")


def export_yaml_to_path(yaml_text: str, path_text: str) -> str:
    if not yaml_text.strip():
        raise ValueError("YAML is empty.")

    path = Path(path_text.strip())
    if not path.name:
        raise ValueError("Export path is required.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text, encoding="utf-8")
    return str(path)


def export_history_entry_to_path(output_name: str, path_text: str) -> str:
    entry = get_history_entry(output_name)
    path = Path(path_text.strip())
    if not path.name:
        raise ValueError("Export path is required.")

    path.parent.mkdir(parents=True, exist_ok=True)
    export_recipe(entry, str(path))
    return str(path)


def build_yaml_download_payload(
    yaml_text: str,
    output_name: str | None = None,
) -> tuple[str, str]:
    if not yaml_text.strip():
        raise ValueError("YAML is empty.")

    raw_base_name = Path((output_name or "").strip()).stem or "merge_recipe"
    base_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_base_name).strip("._")
    if not base_name:
        base_name = "merge_recipe"
    return yaml_text, f"{base_name}.yaml"


def rerun_history_entry(output_name: str) -> str:
    entry = get_history_entry(output_name)

    config = deepcopy(entry.get("config") or {})
    if not config:
        raise ValueError("Selected history entry does not contain a runnable config.")

    task_id, _ = queue_history_yaml(
        history_to_yaml({"config": config, "output_name": output_name}),
        fallback_output_name=output_name,
    )
    return task_id
