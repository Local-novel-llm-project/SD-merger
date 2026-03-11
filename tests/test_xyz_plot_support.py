import pytest

from module.xyz_plot_support import build_axis_preview, parse_axis_values


def test_parse_axis_values_supports_float_range_syntax():
    values = parse_axis_values("Velocity", "0.2:0.6:0.2")

    assert values == ["0.2", "0.4", "0.6"]


def test_parse_axis_values_supports_integer_range_syntax():
    values = parse_axis_values("Steps", "20:40:10")

    assert values == ["20", "30", "40"]


def test_parse_axis_values_rejects_invalid_model_order():
    with pytest.raises(ValueError):
        parse_axis_values("Model Order", "A->B, invalid")


def test_build_axis_preview_reports_total_render_count():
    preview = build_axis_preview(
        "Velocity",
        "0.25, 0.5, 0.75",
        "Strategy",
        "mix, addition",
        "None",
        "",
    )

    assert "Total renders: 6" in preview
