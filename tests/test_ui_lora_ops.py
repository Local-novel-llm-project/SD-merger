from ui.components import lora_ops
import gradio as gr


def test_sync_lora_ratio_map_preserves_existing_ratios_and_defaults_new_entries():
    synced_ratio_map = lora_ops._sync_lora_ratio_map(
        ["style_a.safetensors", "style_b.safetensors"],
        {"style_a.safetensors": 0.4},
    )

    assert synced_ratio_map == {
        "style_a.safetensors": 0.4,
        "style_b.safetensors": 1.0,
    }


def test_sync_lora_ratio_map_drops_removed_entries():
    synced_ratio_map = lora_ops._sync_lora_ratio_map(
        ["style_b.safetensors"],
        {
            "style_a.safetensors": 0.4,
            "style_b.safetensors": 0.9,
        },
    )

    assert synced_ratio_map == {
        "style_b.safetensors": 0.9,
    }


def test_create_lora_ratio_input_is_explicitly_interactive():
    with gr.Blocks():
        ratio_input = lora_ops._create_lora_ratio_input(
            "style_a.safetensors",
            0.4,
            "merge",
        )

    assert ratio_input.interactive is True
    assert ratio_input.step == 0.01


def test_parse_compact_lora_ratio_text_supports_compact_spec_input():
    model_names, ratio_map, error = lora_ops._parse_compact_lora_ratio_text(
        "style_a.safetensors:0.4, style_b.safetensors:0.9",
    )

    assert error is None
    assert model_names == [
        "style_a.safetensors",
        "style_b.safetensors",
    ]
    assert ratio_map == {
        "style_a.safetensors": 0.4,
        "style_b.safetensors": 0.9,
    }


def test_parse_compact_lora_ratio_text_rejects_malformed_compact_spec():
    model_names, ratio_map, error = lora_ops._parse_compact_lora_ratio_text(
        "style_a.safetensors:abc",
    )

    assert model_names is None
    assert ratio_map is None
    assert error == lora_ops.INVALID_RATIO_FORMAT_MESSAGE


def test_import_compact_lora_ratio_text_updates_models_and_ratio_map():
    model_names, ratio_map, status = lora_ops._import_compact_lora_ratio_text(
        "style_a.safetensors:0.4, style_b.safetensors:0.9",
        ["old_style.safetensors"],
        {"old_style.safetensors": 0.2},
    )

    assert model_names == [
        "style_a.safetensors",
        "style_b.safetensors",
    ]
    assert ratio_map == {
        "style_a.safetensors": 0.4,
        "style_b.safetensors": 0.9,
    }
    assert status == "Imported 2 LoRA ratio(s)."


def test_import_compact_lora_ratio_text_preserves_current_state_on_error():
    model_names, ratio_map, status = lora_ops._import_compact_lora_ratio_text(
        "style_a.safetensors:abc",
        ["old_style.safetensors"],
        {"old_style.safetensors": 0.2},
    )

    assert model_names == ["old_style.safetensors"]
    assert ratio_map == {"old_style.safetensors": 0.2}
    assert status == lora_ops.INVALID_RATIO_FORMAT_MESSAGE


def test_resolve_lora_models_and_ratios_uses_selected_model_order(monkeypatch):
    monkeypatch.setattr(
        lora_ops,
        "get_model_path",
        lambda model_name: f"/models/{model_name}",
    )

    model_paths, ratio_list, error = lora_ops._resolve_lora_models_and_ratios(
        ["style_b.safetensors", "style_a.safetensors"],
        {
            "style_a.safetensors": 0.4,
            "style_b.safetensors": 0.9,
        },
    )

    assert error is None
    assert model_paths == [
        "/models/style_b.safetensors",
        "/models/style_a.safetensors",
    ]
    assert ratio_list == [0.9, 0.4]


def test_resolve_lora_models_and_ratios_defaults_missing_ratios(monkeypatch):
    monkeypatch.setattr(
        lora_ops,
        "get_model_path",
        lambda model_name: f"/models/{model_name}",
    )

    model_paths, ratio_list, error = lora_ops._resolve_lora_models_and_ratios(
        ["style_a.safetensors"],
        {},
    )

    assert error is None
    assert model_paths == ["/models/style_a.safetensors"]
    assert ratio_list == [1.0]
