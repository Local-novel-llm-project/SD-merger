import os
import uuid
import logging

import pytest

import main


def _patch_merge_pipeline_dependencies(monkeypatch, calls: dict):
    def fake_model(path):
        calls["model_path"] = path
        return f"recipe::{path}"

    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path

    def fake_load_model(path, lazy_load=True, use_sdxl_keys=None):
        return FakeSDKeyWrapper(path)

    monkeypatch.setattr("module.utility.load_model", fake_load_model)

    def fake_merge(recipe, output):
        calls["merged_recipe"] = recipe
        calls["output"] = output

    def fake_pre_merge(config, recipe):
        calls["pre_merge_config"] = config
        calls["pre_merge_recipe"] = recipe
        return f"tuned::{recipe}"

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)
    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path
    monkeypatch.setattr("module.utility.load_model", lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path))
    monkeypatch.setattr(main.sd_mecha, "merge", fake_merge)
    monkeypatch.setattr(main, "run_pre_merge_hooks", fake_pre_merge)


def test_run_merge_pipeline_allows_target_only_tuning(monkeypatch):
    calls = {}
    _patch_merge_pipeline_dependencies(monkeypatch, calls)

    temp_dir = os.path.join(os.getcwd(), ".pytest_tmp_local", str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)

    output_path = main.run_merge_pipeline(
        {
            "target_model": "models/base_model.safetensors",
            "models": [],
            "output_name": "arthemy_tuned.safetensors",
            "arthemy_tuner": {
                "mode": "Soft Value",
                "unet": {"OUT_Shadows_Depth": 1.1},
            },
        },
        default_output_dir=temp_dir,
    )

    expected_output = os.path.join(temp_dir, "arthemy_tuned.safetensors")
    assert output_path == expected_output
    assert calls["model_path"] == "models/base_model.safetensors"
    assert calls["pre_merge_recipe"] == "recipe::models/base_model.safetensors"
    assert calls["merged_recipe"] == "tuned::recipe::models/base_model.safetensors"
    assert calls["output"] == expected_output


def test_run_merge_pipeline_requires_models_or_target_model():
    with pytest.raises(main.ConfigError):
        main.run_merge_pipeline({"models": []}, default_output_dir=os.getcwd())


def test_run_merge_pipeline_uses_config_output_dir_when_explicit(monkeypatch):
    calls = {}
    _patch_merge_pipeline_dependencies(monkeypatch, calls)

    default_output_dir = os.path.join(os.getcwd(), ".pytest_tmp_local", "default")
    configured_output_dir = os.path.join(os.getcwd(), ".pytest_tmp_local", "configured")

    output_path = main.run_merge_pipeline(
        {
            "target_model": "models/base_model.safetensors",
            "models": [],
            "output_dir": configured_output_dir,
            "output_name": "configured_output.safetensors",
        },
        default_output_dir=default_output_dir,
    )

    expected_output = os.path.join(configured_output_dir, "configured_output.safetensors")
    assert output_path == expected_output
    assert calls["output"] == expected_output


def test_run_merge_pipeline_rejects_non_mapping_config():
    with pytest.raises(main.ConfigError):
        main.run_merge_pipeline(["not", "a", "mapping"], default_output_dir=os.getcwd())


def test_run_merge_pipeline_preserves_debug_sd_mecha_log_level(monkeypatch):
    calls = {}
    _patch_merge_pipeline_dependencies(monkeypatch, calls)

    observed_levels = []
    monkeypatch.setattr(
        main.sd_mecha,
        "set_log_level",
        lambda level: observed_levels.append(level),
    )
    monkeypatch.setattr(
        main.logger,
        "isEnabledFor",
        lambda level: level == logging.DEBUG,
    )

    main.run_merge_pipeline(
        {
            "target_model": "models/base_model.safetensors",
            "models": [],
            "output_name": "debug_output.safetensors",
        },
        default_output_dir=os.getcwd(),
    )

    assert observed_levels[-1] == logging.DEBUG


def test_run_merge_pipeline_uses_default_target_strategy_when_omitted(monkeypatch):
    calls = {}

    def fake_model(path):
        return f"recipe::{path}"

    def fake_calc_strategy(strategy_name, replace_with):
        calls["calc_strategy"] = (strategy_name, replace_with)

        def runner(left_node, right_node, velocity, key_patterns_json):
            calls["diff_args"] = (left_node, right_node, velocity, key_patterns_json)
            return "diff-node"

        return runner

    def fake_target_strategy(strategy_name):
        calls["target_strategy"] = strategy_name
        return lambda *args, **kwargs: "unused-target-node"

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)
    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path
    monkeypatch.setattr("module.utility.load_model", lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path))
    monkeypatch.setattr(main.sd_mecha, "merge", lambda recipe, output: calls.update({"output": output, "merged_recipe": recipe}))
    monkeypatch.setattr(main, "get_calculation_strategy", fake_calc_strategy)
    monkeypatch.setattr(main, "get_target_calculation_strategy", fake_target_strategy)
    monkeypatch.setattr(main, "run_pre_merge_hooks", lambda config, recipe: recipe)
    monkeypatch.setattr(main, "scale_tensor", lambda diff_node, scale: f"scaled::{diff_node}::{scale}")

    output_path = main.run_merge_pipeline(
        {
            "models": [
                {
                    "left": "models/a.safetensors",
                    "right": "models/b.safetensors",
                    "key_patterns": ["."],
                }
            ],
            "output_name": "default_target_strategy.safetensors",
        },
        default_output_dir=os.getcwd(),
    )

    assert calls["calc_strategy"] == ("addition", None)
    assert calls["target_strategy"] == "addition"
    assert calls["merged_recipe"] == "scaled::diff-node::0.5"
    assert output_path.endswith("default_target_strategy.safetensors")
