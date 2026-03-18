from module.lora_input import (
    is_compact_lora_spec_text,
    normalize_lora_models_and_ratios,
)


def test_normalize_lora_models_and_ratios_supports_compact_string_specs():
    models, ratios = normalize_lora_models_and_ratios(
        "style_a.safetensors:0.4, style_b.safetensors:0.9"
    )

    assert models == ["style_a.safetensors", "style_b.safetensors"]
    assert ratios == [0.4, 0.9]


def test_normalize_lora_models_and_ratios_keeps_windows_paths_without_inline_ratio():
    models, ratios = normalize_lora_models_and_ratios(
        [r"C:\loras\style_a.safetensors", r"D:\loras\style_b.safetensors"],
        [0.6, 0.2],
    )

    assert models == [
        r"C:\loras\style_a.safetensors",
        r"D:\loras\style_b.safetensors",
    ]
    assert ratios == [0.6, 0.2]


def test_normalize_lora_models_and_ratios_prioritizes_inline_ratio_over_ratio_list():
    models, ratios = normalize_lora_models_and_ratios(
        ["style_a.safetensors:0.4", "style_b.safetensors"],
        [0.8, 0.2],
    )

    assert models == ["style_a.safetensors", "style_b.safetensors"]
    assert ratios == [0.4, 0.2]


def test_is_compact_lora_spec_text_detects_model_ratio_pairs():
    assert is_compact_lora_spec_text("style_a.safetensors:0.4, style_b.safetensors:0.9")
    assert not is_compact_lora_spec_text("0.4, 0.9")
