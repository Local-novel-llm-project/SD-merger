from __future__ import annotations

from typing import Any


def get_primary_model_config(config: dict[str, Any]) -> dict[str, Any]:
    models = config.get("models")
    if not isinstance(models, list):
        return {}
    for model in models:
        if isinstance(model, dict):
            return model
    return {}


def resolve_left_right_velocity(config: dict[str, Any], model_config: dict[str, Any]):
    for key in ("left_right_velocity", "lrv", "lr", "strategy_velocity"):
        value = model_config.get(key)
        if value not in (None, ""):
            return value

    velocity = model_config.get("velocity", "")
    if velocity in (None, ""):
        return ""

    if config.get("target_model"):
        return 1.0

    return velocity


def resolve_output_name(
    config: dict[str, Any],
    default_output_name: str = "imported_recipe_merge.safetensors",
) -> str:
    output_name = config.get("output_name")
    if output_name:
        return str(output_name)

    model_config = get_primary_model_config(config)
    if model_config.get("output_name"):
        return str(model_config["output_name"])

    return default_output_name
