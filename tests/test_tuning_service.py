import ui.services.tuning_service as tuning_service


def test_parse_optional_float_returns_none_for_blank():
    assert tuning_service.parse_optional_float("   ") is None


def test_build_tuning_preview_returns_hint_when_model_missing():
    preview = tuning_service.build_tuning_preview(
        "",
        "Soft Value",
        "1.0",
        "1.0",
        "",
        "arthemy_tuned.safetensors",
    )

    assert "hint" in preview


def test_build_tuning_preview_returns_error_json(monkeypatch):
    monkeypatch.setattr(
        tuning_service,
        "build_tuning_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("broken config")),
    )

    preview = tuning_service.build_tuning_preview(
        "model.safetensors",
        "Soft Value",
        "1.0",
        "1.0",
        "",
        "arthemy_tuned.safetensors",
    )

    assert "broken config" in preview
