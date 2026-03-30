import module.services.merge as merge_services
from module.services.merge import config_builder
import ui.services.merge_service as merge_service


def test_build_basic_merge_config_omits_left_right_velocity_when_blank(monkeypatch):
    monkeypatch.setattr(
        config_builder,
        "resolve_model_path",
        lambda name: f"/models/{name}" if name else None,
    )

    config, output_name = config_builder.build_basic_merge_config(
        "ModelA",
        "ModelB",
        "選択しない",
        "mix",
        "mix",
        0.5,
        "",
        False,
        "",
        "",
        "",
        True,
    )

    assert "target_model" not in config
    assert config["models"][0]["left"] == "/models/ModelA"
    assert config["models"][0]["right"] == "/models/ModelB"
    assert config["models"][0]["velocity"] == 0.5
    assert "left_right_velocity" not in config["models"][0]
    assert output_name


def test_build_basic_merge_config_includes_optional_fields(monkeypatch):
    monkeypatch.setattr(
        config_builder,
        "resolve_model_path",
        lambda name: f"/models/{name}" if name else None,
    )

    config, output_name = config_builder.build_basic_merge_config(
        "ModelA",
        "ModelB",
        "ModelC",
        "subtraction",
        "addition",
        0.25,
        "0.8",
        True,
        "1,1,1",
        "vae.safetensors",
        "merged.safetensors",
        False,
    )

    assert config["target_model"] == "/models/ModelC"
    assert config["models"][0]["left_right_velocity"] == 0.8
    assert config["models"][0]["mbw"] == "1,1,1"
    assert config["bake_in_vae"] == "/models/vae.safetensors"
    assert config["output_name"] == "merged.safetensors"
    assert output_name == "merged.safetensors"


def test_build_merge_preview_returns_hint_when_models_missing():
    preview = config_builder.build_merge_preview(
        "",
        "",
        "選択しない",
        "mix",
        "mix",
        "0.5",
        "",
        False,
        "",
        "",
        "",
        True,
    )

    assert "hint" in preview


def test_build_merge_preview_returns_error_json(monkeypatch):
    monkeypatch.setattr(
        config_builder,
        "build_basic_merge_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad input")),
    )

    preview = config_builder.build_merge_preview(
        "ModelA",
        "ModelB",
        "選択しない",
        "mix",
        "mix",
        "0.5",
        "",
        False,
        "",
        "",
        "",
        True,
    )

    assert "bad input" in preview


def test_ui_merge_service_reexports_new_helpers():
    assert merge_service.build_basic_merge_config is merge_services.build_basic_merge_config
    assert merge_service.build_merge_preview is merge_services.build_merge_preview
    assert merge_service.queue_merge is merge_services.queue_merge
