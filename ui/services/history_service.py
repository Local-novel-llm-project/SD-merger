from __future__ import annotations

import os
from copy import deepcopy

import yaml

from module.history import export_recipe, history_to_yaml, load_history
from ui.services.execution_service import enqueue_merge_task


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


def find_history_entry(output_name: str) -> dict | None:
    for entry in load_history():
        if str(entry.get("output_name") or "") == output_name:
            return entry
    return None


def build_history_yaml(output_name: str) -> str:
    entry = find_history_entry(output_name)
    if entry is None:
        raise ValueError(f"History entry was not found: {output_name}")
    return history_to_yaml(entry)


def export_history_recipe(output_name: str, filepath: str) -> None:
    entry = find_history_entry(output_name)
    if entry is None:
        raise ValueError(f"History entry was not found: {output_name}")
    export_recipe(entry, filepath)


def import_recipe_yaml(yaml_text: str) -> dict:
    loaded = yaml.safe_load(yaml_text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Recipe YAML must contain a mapping at the top level.")
    return loaded


def rerun_history_entry(output_name: str) -> str:
    entry = find_history_entry(output_name)
    if entry is None:
        raise ValueError(f"History entry was not found: {output_name}")

    config = deepcopy(entry.get("config") or {})
    if not config:
        raise ValueError("Selected history entry does not contain a runnable config.")

    return enqueue_merge_task(
        config,
        output_name,
        task_name=f"Rerun: {output_name}",
    )
