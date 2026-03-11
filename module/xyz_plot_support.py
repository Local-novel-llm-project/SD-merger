from __future__ import annotations

from typing import List


AXIS_DEFAULT_VALUES = {
    "Velocity": "0.25, 0.5, 0.75",
    "Strategy": "mix, addition",
    "CFG Scale": "5, 7, 9",
    "Steps": "20, 30, 40",
    "Model Order": "A->B, B->A",
    "None": "",
}

NUMERIC_AXES = {"Velocity", "CFG Scale", "Steps"}
INTEGER_AXES = {"Steps"}
VALID_MODEL_ORDERS = {"A->B", "B->A"}


def get_axis_default_values(axis_type: str) -> str:
    return AXIS_DEFAULT_VALUES.get(axis_type, "")


def get_axis_help_text(axis_type: str) -> str:
    if axis_type == "None":
        return "No values required for this axis."

    example = get_axis_default_values(axis_type)
    if axis_type in NUMERIC_AXES:
        return (
            f"Use comma-separated values or `start:end:step`. "
            f"Example: `{example}` or `0.2:0.8:0.2`."
        )
    if axis_type == "Model Order":
        return f"Allowed values: `{example}`."
    return f"Use comma-separated values. Example: `{example}`."


def _format_numeric_value(value: float, *, as_int: bool) -> str:
    if as_int:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _expand_numeric_range(axis_type: str, raw_values: str) -> List[str]:
    parts = [part.strip() for part in raw_values.split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"{axis_type} range must use `start:end:step` format."
        )

    try:
        start, end, step = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{axis_type} range values must be numeric.") from exc

    if step == 0:
        raise ValueError(f"{axis_type} range step must not be zero.")

    if axis_type in INTEGER_AXES and any(not value.is_integer() for value in (start, end, step)):
        raise ValueError(f"{axis_type} range must use integer values.")

    if (end - start) * step < 0:
        raise ValueError(
            f"{axis_type} range step direction does not reach the end value."
        )

    values = []
    current = start
    epsilon = abs(step) / 1000 or 1e-9
    as_int = axis_type in INTEGER_AXES

    if step > 0:
        while current <= end + epsilon:
            values.append(_format_numeric_value(current, as_int=as_int))
            current += step
    else:
        while current >= end - epsilon:
            values.append(_format_numeric_value(current, as_int=as_int))
            current += step

    if not values:
        raise ValueError(f"{axis_type} range did not produce any values.")

    return values


def parse_axis_values(axis_type: str, raw_values: str) -> List[str]:
    if axis_type == "None":
        return ["None"]

    cleaned = (raw_values or "").strip()
    if not cleaned:
        raise ValueError(f"{axis_type} requires at least one value.")

    if axis_type in NUMERIC_AXES and ":" in cleaned and "," not in cleaned:
        values = _expand_numeric_range(axis_type, cleaned)
    else:
        values = [value.strip() for value in cleaned.split(",") if value.strip()]

    if not values:
        raise ValueError(f"{axis_type} requires at least one value.")

    if axis_type in {"Velocity", "CFG Scale"}:
        for value in values:
            try:
                float(value)
            except ValueError as exc:
                raise ValueError(f"{axis_type} values must be numeric.") from exc

    if axis_type == "Steps":
        for value in values:
            try:
                int(value)
            except ValueError as exc:
                raise ValueError("Steps values must be integers.") from exc

    if axis_type == "Model Order":
        invalid_values = [value for value in values if value not in VALID_MODEL_ORDERS]
        if invalid_values:
            raise ValueError("Model Order must be `A->B` or `B->A`.")

    return values


def build_axis_preview(
    x_type: str,
    x_values: str,
    y_type: str,
    y_values: str,
    z_type: str,
    z_values: str,
) -> str:
    try:
        parsed_x = parse_axis_values(x_type, x_values)
        parsed_y = parse_axis_values(y_type, y_values)
        parsed_z = parse_axis_values(z_type, z_values)
    except ValueError as exc:
        return f"Input preview: {exc}"

    total = len(parsed_x) * len(parsed_y) * len(parsed_z)
    preview_lines = [
        f"X: {', '.join(parsed_x)}",
        f"Y: {', '.join(parsed_y)}",
        f"Z: {', '.join(parsed_z)}",
        f"Total renders: {total}",
    ]

    if total > 24:
        preview_lines.append("Warning: large grids can take a long time to generate.")

    return "\n".join(preview_lines)
