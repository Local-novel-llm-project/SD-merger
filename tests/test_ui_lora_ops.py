from ui.components import lora_ops


def test_parse_lora_models_and_ratios_supports_compact_spec_input(monkeypatch):
    monkeypatch.setattr(
        lora_ops,
        "get_model_path",
        lambda model_name: f"/models/{model_name}",
    )

    model_paths, ratio_list, error = lora_ops._parse_lora_models_and_ratios(
        [],
        "style_a.safetensors:0.4, style_b.safetensors:0.9",
    )

    assert error is None
    assert model_paths == [
        "/models/style_a.safetensors",
        "/models/style_b.safetensors",
    ]
    assert ratio_list == [0.4, 0.9]


def test_parse_lora_models_and_ratios_uses_selected_models_for_plain_ratio_input(monkeypatch):
    monkeypatch.setattr(
        lora_ops,
        "get_model_path",
        lambda model_name: f"/models/{model_name}",
    )

    model_paths, ratio_list, error = lora_ops._parse_lora_models_and_ratios(
        ["style_a.safetensors", "style_b.safetensors"],
        "0.4, 0.9",
    )

    assert error is None
    assert model_paths == [
        "/models/style_a.safetensors",
        "/models/style_b.safetensors",
    ]
    assert ratio_list == [0.4, 0.9]


def test_parse_lora_models_and_ratios_rejects_malformed_compact_spec():
    model_paths, ratio_list, error = lora_ops._parse_lora_models_and_ratios(
        [],
        "style_a.safetensors:abc",
    )

    assert model_paths is None
    assert ratio_list is None
    assert error == (
        "Invalid format. Use '0.5, 1.0' or "
        "'lora_a.safetensors:0.5, lora_b.safetensors:1.0'."
    )
