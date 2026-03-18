from __future__ import annotations

from collections.abc import Sequence


def _split_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_model_entries(models: Sequence[str] | str | None) -> list[str]:
    if models is None:
        return []
    if isinstance(models, str):
        return _split_csv_values(models)

    entries: list[str] = []
    for model in models:
        model_text = str(model).strip()
        if model_text:
            entries.append(model_text)
    return entries


def _parse_inline_ratio(model_entry: str) -> tuple[str, float | None]:
    if ":" not in model_entry:
        return model_entry, None

    maybe_model, maybe_ratio = model_entry.rsplit(":", 1)
    model_name = maybe_model.strip()
    ratio_text = maybe_ratio.strip()
    if not model_name or not ratio_text:
        return model_entry, None

    try:
        return model_name, float(ratio_text)
    except ValueError:
        return model_entry, None


def _normalize_ratio_entries(
    ratios: Sequence[float] | str | None,
    model_count: int,
    default_ratio: float,
) -> list[float]:
    if ratios is None:
        normalized_ratios: list[float] = []
    elif isinstance(ratios, str):
        normalized_ratios = [float(item) for item in _split_csv_values(ratios)]
    else:
        normalized_ratios = [float(value) for value in ratios]

    if len(normalized_ratios) < model_count:
        normalized_ratios.extend([default_ratio] * (model_count - len(normalized_ratios)))
    elif len(normalized_ratios) > model_count:
        normalized_ratios = normalized_ratios[:model_count]

    return normalized_ratios


def is_compact_lora_spec_text(value: str | None) -> bool:
    if not value or not value.strip():
        return False

    for entry in _split_csv_values(value):
        _, inline_ratio = _parse_inline_ratio(entry)
        if inline_ratio is not None:
            return True
    return False


def normalize_lora_models_and_ratios(
    models: Sequence[str] | str | None,
    ratios: Sequence[float] | str | None = None,
    *,
    default_ratio: float = 1.0,
) -> tuple[list[str], list[float]]:
    model_entries = _normalize_model_entries(models)
    normalized_ratios = _normalize_ratio_entries(ratios, len(model_entries), default_ratio)

    normalized_models: list[str] = []
    resolved_ratios: list[float] = []

    for index, model_entry in enumerate(model_entries):
        model_name, inline_ratio = _parse_inline_ratio(model_entry)
        normalized_models.append(model_name)
        resolved_ratios.append(
            inline_ratio if inline_ratio is not None else normalized_ratios[index]
        )

    return normalized_models, resolved_ratios
