from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


SD15_KEYWORDS = [
    "cond_stage_model",
    "model.diffusion_model.input_blocks.0.",
    "model.diffusion_model.input_blocks.1.",
    "model.diffusion_model.input_blocks.2.",
    "model.diffusion_model.input_blocks.3.",
    "model.diffusion_model.input_blocks.4.",
    "model.diffusion_model.input_blocks.5.",
    "model.diffusion_model.input_blocks.6.",
    "model.diffusion_model.input_blocks.7.",
    "model.diffusion_model.input_blocks.8.",
    "model.diffusion_model.input_blocks.9.",
    "model.diffusion_model.input_blocks.10.",
    "model.diffusion_model.input_blocks.11.",
    "model.diffusion_model.middle_block",
    "model.diffusion_model.output_blocks.0.",
    "model.diffusion_model.output_blocks.1.",
    "model.diffusion_model.output_blocks.2.",
    "model.diffusion_model.output_blocks.3.",
    "model.diffusion_model.output_blocks.4.",
    "model.diffusion_model.output_blocks.5.",
    "model.diffusion_model.output_blocks.6.",
    "model.diffusion_model.output_blocks.7.",
    "model.diffusion_model.output_blocks.8.",
    "model.diffusion_model.output_blocks.9.",
    "model.diffusion_model.output_blocks.10.",
    "model.diffusion_model.output_blocks.11.",
]

SDXL_KEYWORDS = [
    "conditioner",
    "model.diffusion_model.input_blocks.0.",
    "model.diffusion_model.input_blocks.1.",
    "model.diffusion_model.input_blocks.2.",
    "model.diffusion_model.input_blocks.3.",
    "model.diffusion_model.input_blocks.4.",
    "model.diffusion_model.input_blocks.5.",
    "model.diffusion_model.input_blocks.6.",
    "model.diffusion_model.input_blocks.7.",
    "model.diffusion_model.input_blocks.8.",
    "model.diffusion_model.middle_block",
    "model.diffusion_model.output_blocks.0.",
    "model.diffusion_model.output_blocks.1.",
    "model.diffusion_model.output_blocks.2.",
    "model.diffusion_model.output_blocks.3.",
    "model.diffusion_model.output_blocks.4.",
    "model.diffusion_model.output_blocks.5.",
    "model.diffusion_model.output_blocks.6.",
    "model.diffusion_model.output_blocks.7.",
    "model.diffusion_model.output_blocks.8.",
]


@dataclass(frozen=True)
class CurveWeightsResult:
    profile: str
    weights: tuple[float, ...]
    mbw_a: str
    mbw_b: str


def parse_ratio_string(raw_value: str) -> list[float]:
    return [float(part.strip()) for part in raw_value.split(",")]


def resolve_block_keywords(block_count: int) -> list[str]:
    if block_count == len(SD15_KEYWORDS):
        return list(SD15_KEYWORDS)
    if block_count == len(SDXL_KEYWORDS):
        return list(SDXL_KEYWORDS)
    raise ValueError(f"MBW Each expects {len(SD15_KEYWORDS)} or {len(SDXL_KEYWORDS)} blocks, got {block_count}.")


def _curve_position(index: int, total: int) -> float:
    if total <= 1:
        return 0.0
    return index / (total - 1)


def build_curve_weights(
    *,
    block_count: int,
    start: float = 0.0,
    end: float = 1.0,
    profile: str = "linear",
) -> CurveWeightsResult:
    profile_name = profile.strip().lower()
    weights: list[float] = []

    for index in range(block_count):
        t = _curve_position(index, block_count)
        if profile_name == "linear":
            value = start + (end - start) * t
        elif profile_name == "ease_in":
            value = start + (end - start) * (t * t)
        elif profile_name == "ease_out":
            value = start + (end - start) * (1 - (1 - t) * (1 - t))
        elif profile_name == "ease_in_out":
            smooth = 0.5 - 0.5 * math.cos(math.pi * t)
            value = start + (end - start) * smooth
        else:
            raise ValueError(f"Unsupported curve profile: {profile}")
        weights.append(value)

    return CurveWeightsResult(
        profile=profile_name,
        weights=tuple(weights),
        mbw_a=format_mbw_values(1.0 - weight for weight in weights),
        mbw_b=format_mbw_values(weights),
    )


def format_mbw_values(values: Iterable[float]) -> str:
    return ",".join(f"{float(value):.6f}" for value in values)


def build_key_pattern_rules(ratios_a: list[float], ratios_b: list[float]) -> dict[str, dict[str, Any]]:
    if len(ratios_a) != len(ratios_b):
        raise ValueError("mbw_a and mbw_b must contain the same number of values.")

    keywords = resolve_block_keywords(len(ratios_a))
    return {
        f"block_{index}": {"pattern": pattern, "a": ratio_a, "b": ratio_b}
        for index, (ratio_a, ratio_b, pattern) in enumerate(zip(ratios_a, ratios_b, keywords))
    }


def build_mbw_each_model_config(
    model_entry: dict[str, Any],
) -> dict[str, Any]:
    mbw_a = str(model_entry.get("mbw_a", "")).strip()
    mbw_b = str(model_entry.get("mbw_b", "")).strip()
    if not mbw_a or not mbw_b:
        return dict(model_entry)

    try:
        ratios_a = parse_ratio_string(mbw_a)
        ratios_b = parse_ratio_string(mbw_b)
    except ValueError as exc:
        raise ValueError("MBW Each values must be comma-separated numbers.") from exc

    rules = build_key_pattern_rules(ratios_a, ratios_b)
    updated_entry = dict(model_entry)
    updated_entry["key_patterns"] = rules
    updated_entry.pop("mbw_a", None)
    updated_entry.pop("mbw_b", None)
    return updated_entry


def build_mbw_each_config(config: dict[str, Any]) -> dict[str, Any]:
    updated_config = dict(config)
    updated_config["models"] = [
        build_mbw_each_model_config(model_entry) for model_entry in config.get("models", [])
    ]
    return updated_config
