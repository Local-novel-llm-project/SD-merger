from __future__ import annotations

import json
from typing import Any

from module.utility import generate_filename

from .model_catalog import resolve_model_path
from .queue_tasks import enqueue_merge_task

MERGE_STRATEGIES = [
    "addition",
    "subtraction",
    "multiplication",
    "mix",
    "cosineA",
    "cosineB",
    "smoothAdd",
    "tensor",
    "tensor2",
    "mbw_each",
    "quantum",
]

TARGET_STRATEGIES = [
    "mix",
    "addition",
    "subtraction",
    "angle",
    "trainDifference",
    "extract",
]

MERGE_VELOCITY_HELP = (
    "`Velocity` は最終適用量です。"
    " `LRV` は A/B を計算する段階の量で、"
    " target/base へ適用する前の混ぜ方を変えます。"
)


def create_default_output_name(model_a_name: str, model_b_name: str) -> str:
    return generate_filename(model_a_name, model_b_name)


def parse_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc


def build_basic_merge_config(
    model_a: str,
    model_b: str,
    model_c: str | None,
    strategy: str,
    target_strategy: str,
    velocity: float,
    left_right_velocity: str | None,
    use_advanced_options: bool,
    mbw: str | None,
    bake_in_vae: str | None,
    output_name: str | None,
    lazy_load: bool,
) -> tuple[dict[str, Any], str]:
    if not model_a or not model_b:
        raise ValueError("Model A and Model B are required.")

    resolved_target_name = model_c if model_c and model_c != "選択しない" else None
    target_model_path = (
        resolve_model_path(resolved_target_name) if resolved_target_name else None
    )
    left_model_path = resolve_model_path(model_a)
    right_model_path = resolve_model_path(model_b)

    if not left_model_path or not right_model_path:
        raise ValueError("Selected models could not be resolved from the models directory.")
    if resolved_target_name and not target_model_path:
        raise ValueError("Selected models could not be resolved from the models directory.")

    primary_model_config: dict[str, Any] = {
        "left": left_model_path,
        "right": right_model_path,
        "strategy": strategy,
        "target_strategy": target_strategy,
        "velocity": float(velocity),
        "key_patterns": ["."],
    }
    config: dict[str, Any] = {
        "lazy_load": lazy_load,
        "models": [primary_model_config],
    }
    if target_model_path:
        config["target_model"] = target_model_path

    if use_advanced_options and mbw and mbw.strip():
        primary_model_config["mbw"] = mbw.strip()

    if use_advanced_options:
        parsed_lrv = parse_optional_float(
            left_right_velocity,
            field_name="A/B Strategy Velocity",
        )
        if parsed_lrv is not None:
            primary_model_config["left_right_velocity"] = parsed_lrv

    if use_advanced_options and bake_in_vae:
        vae_path = resolve_model_path(bake_in_vae)
        if vae_path:
            config["bake_in_vae"] = vae_path

    explicit_output_name = (output_name or "").strip()
    resolved_output_name = (
        explicit_output_name
        if use_advanced_options and explicit_output_name
        else create_default_output_name(model_a, model_b)
    )
    if use_advanced_options and explicit_output_name:
        config["output_name"] = resolved_output_name

    return config, resolved_output_name


def build_preview_json(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=False)


def build_merge_preview(
    model_a: str,
    model_b: str,
    model_c: str | None,
    strategy: str,
    target_strategy: str,
    velocity: str,
    left_right_velocity: str | None,
    use_advanced_options: bool,
    mbw: str | None,
    bake_in_vae: str | None,
    output_name: str | None,
    lazy_load: bool,
) -> str:
    if not model_a or not model_b:
        return json.dumps(
            {"hint": "Model A と Model B を選ぶと設定プレビューを表示します。"},
            indent=2,
            ensure_ascii=False,
        )

    try:
        config, _ = build_basic_merge_config(
            model_a,
            model_b,
            model_c,
            strategy,
            target_strategy,
            float(velocity),
            left_right_velocity,
            use_advanced_options,
            mbw,
            bake_in_vae,
            output_name,
            lazy_load,
        )
        return build_preview_json(config)
    except Exception as exc:
        return json.dumps(
            {"error": str(exc)},
            indent=2,
            ensure_ascii=False,
        )


def queue_merge(
    config: dict[str, Any],
    output_name: str,
    *,
    task_name: str = "Merge Models",
) -> str:
    return enqueue_merge_task(config, output_name, task_name)
