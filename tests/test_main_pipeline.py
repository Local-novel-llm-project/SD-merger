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
        def runner(target_node, diff_node, velocity, key_patterns_json):
            calls["target_args"] = (target_node, diff_node, velocity, key_patterns_json)
            return f"merged::{target_node}::{diff_node}::{velocity}"

        return runner

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)
    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path
    monkeypatch.setattr("module.utility.load_model", lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path))
    monkeypatch.setattr(main.sd_mecha, "merge", lambda recipe, output: calls.update({"output": output, "merged_recipe": recipe}))
    monkeypatch.setattr(main, "get_calculation_strategy", fake_calc_strategy)
    monkeypatch.setattr(main, "get_target_calculation_strategy", fake_target_strategy)
    monkeypatch.setattr(main, "run_pre_merge_hooks", lambda config, recipe: recipe)

    output_path = main.run_merge_pipeline(
        {
            "target_model": "models/base.safetensors",
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
    assert calls["target_args"] == (
        "recipe::models/base.safetensors",
        "diff-node",
        0.5,
        '["."]',
    )
    assert calls["merged_recipe"] == "merged::recipe::models/base.safetensors::diff-node::0.5"
    assert output_path.endswith("default_target_strategy.safetensors")


def test_run_merge_pipeline_ab_uses_velocity_for_strategy_when_left_right_velocity_omitted(
    monkeypatch,
):
    calls = {}

    def fake_model(path):
        return f"recipe::{path}"

    def fake_calc_strategy(strategy_name, replace_with):
        calls["calc_strategy"] = (strategy_name, replace_with)

        def runner(left_node, right_node, velocity, key_patterns_json):
            calls["diff_args"] = (left_node, right_node, velocity, key_patterns_json)
            return "diff-node"

        return runner

    def unexpected_target_strategy(strategy_name):
        raise AssertionError(
            f"AB マージでは target_strategy を解決しない想定です: {strategy_name}"
        )

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)

    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path

    monkeypatch.setattr(
        "module.utility.load_model",
        lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path),
    )
    monkeypatch.setattr(
        main.sd_mecha,
        "merge",
        lambda recipe, output: calls.update({"output": output, "merged_recipe": recipe}),
    )
    monkeypatch.setattr(main, "get_calculation_strategy", fake_calc_strategy)
    monkeypatch.setattr(main, "get_target_calculation_strategy", unexpected_target_strategy)
    monkeypatch.setattr(main, "run_pre_merge_hooks", lambda config, recipe: recipe)

    output_path = main.run_merge_pipeline(
        {
            "models": [
                {
                    "left": "models/a.safetensors",
                    "right": "models/b.safetensors",
                    "velocity": 0.25,
                    "key_patterns": ["."],
                }
            ],
            "output_name": "ab_velocity_strategy.safetensors",
        },
        default_output_dir=os.getcwd(),
    )

    assert calls["calc_strategy"] == ("addition", None)
    assert calls["diff_args"] == (
        "recipe::models/a.safetensors",
        "recipe::models/b.safetensors",
        0.25,
        '["."]',
    )
    assert calls["merged_recipe"] == "diff-node"
    assert output_path.endswith("ab_velocity_strategy.safetensors")


def test_run_merge_pipeline_ab_honors_explicit_left_right_velocity(monkeypatch):
    calls = {}

    def fake_model(path):
        return f"recipe::{path}"

    def fake_calc_strategy(strategy_name, replace_with):
        def runner(left_node, right_node, velocity, key_patterns_json):
            calls["diff_args"] = (left_node, right_node, velocity, key_patterns_json)
            return "diff-node"

        return runner

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)

    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path

    monkeypatch.setattr(
        "module.utility.load_model",
        lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path),
    )
    monkeypatch.setattr(
        main.sd_mecha,
        "merge",
        lambda recipe, output: calls.update({"output": output, "merged_recipe": recipe}),
    )
    monkeypatch.setattr(main, "get_calculation_strategy", fake_calc_strategy)
    monkeypatch.setattr(main, "run_pre_merge_hooks", lambda config, recipe: recipe)

    output_path = main.run_merge_pipeline(
        {
            "models": [
                {
                    "left": "models/a.safetensors",
                    "right": "models/b.safetensors",
                    "velocity": 0.25,
                    "left_right_velocity": 1.0,
                    "key_patterns": ["."],
                }
            ],
            "output_name": "ab_explicit_lr_velocity.safetensors",
        },
        default_output_dir=os.getcwd(),
    )

    assert calls["diff_args"] == (
        "recipe::models/a.safetensors",
        "recipe::models/b.safetensors",
        1.0,
        '["."]',
    )
    assert calls["merged_recipe"] == "diff-node"
    assert output_path.endswith("ab_explicit_lr_velocity.safetensors")


def test_run_merge_pipeline_passes_velocity_to_angle_target_strategy(monkeypatch):
    calls = {}

    def fake_model(path):
        return f"recipe::{path}"

    def fake_calc_strategy(strategy_name, replace_with):
        def runner(left_node, right_node, velocity, key_patterns_json):
            return f"diff::{left_node}->{right_node}::{velocity}"

        return runner

    def fake_target_strategy(strategy_name):
        assert strategy_name == "angle"

        def runner(target_node, **kwargs):
            calls["target_kwargs"] = (target_node, kwargs)
            return "angle-merged"

        return runner

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)

    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path

    monkeypatch.setattr(
        "module.utility.load_model",
        lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path),
    )
    monkeypatch.setattr(
        main.sd_mecha,
        "merge",
        lambda recipe, output: calls.update({"merged_recipe": recipe, "output": output}),
    )
    monkeypatch.setattr(main, "get_calculation_strategy", fake_calc_strategy)
    monkeypatch.setattr(main, "get_target_calculation_strategy", fake_target_strategy)
    monkeypatch.setattr(main, "run_pre_merge_hooks", lambda config, recipe: recipe)

    main.run_merge_pipeline(
        {
            "target_model": "models/base.safetensors",
            "models": [
                {
                    "left": "models/a.safetensors",
                    "right": "models/b.safetensors",
                    "strategy": "subtraction",
                    "target_strategy": "angle",
                    "velocity": 0.3,
                    "key_patterns": ["."],
                }
            ],
            "output_name": "angle_target.safetensors",
        },
        default_output_dir=os.getcwd(),
    )

    target_node, kwargs = calls["target_kwargs"]
    assert target_node == "recipe::models/base.safetensors"
    assert kwargs["diff_l"] == "diff::recipe::models/a.safetensors->recipe::models/base.safetensors::1.0"
    assert kwargs["diff_r"] == "diff::recipe::models/b.safetensors->recipe::models/base.safetensors::1.0"
    assert kwargs["left"] == "recipe::models/a.safetensors"
    assert kwargs["right"] == "recipe::models/b.safetensors"
    assert kwargs["velocity"] == 0.3
    assert calls["merged_recipe"] == "angle-merged"


def test_run_merge_pipeline_wires_extract_target_strategy_with_left_and_right(monkeypatch):
    calls = {}

    def fake_model(path):
        return f"recipe::{path}"

    def fake_calc_strategy(strategy_name, replace_with):
        def runner(left_node, right_node, velocity, key_patterns_json):
            return "unused-diff"

        return runner

    def fake_target_strategy(strategy_name):
        assert strategy_name == "extract"

        def runner(target_node, **kwargs):
            calls["target_kwargs"] = (target_node, kwargs)
            return "extract-merged"

        return runner

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)

    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path

    monkeypatch.setattr(
        "module.utility.load_model",
        lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path),
    )
    monkeypatch.setattr(
        main.sd_mecha,
        "merge",
        lambda recipe, output: calls.update({"merged_recipe": recipe, "output": output}),
    )
    monkeypatch.setattr(main, "get_calculation_strategy", fake_calc_strategy)
    monkeypatch.setattr(main, "get_target_calculation_strategy", fake_target_strategy)
    monkeypatch.setattr(main, "run_pre_merge_hooks", lambda config, recipe: recipe)

    main.run_merge_pipeline(
        {
            "target_model": "models/base.safetensors",
            "models": [
                {
                    "left": "models/a.safetensors",
                    "right": "models/b.safetensors",
                    "target_strategy": "extract",
                    "velocity": 0.4,
                    "key_patterns": ["."],
                }
            ],
            "output_name": "extract_target.safetensors",
        },
        default_output_dir=os.getcwd(),
    )

    target_node, kwargs = calls["target_kwargs"]
    assert target_node == "recipe::models/base.safetensors"
    assert kwargs["a"] == "recipe::models/a.safetensors"
    assert kwargs["b"] == "recipe::models/b.safetensors"
    assert kwargs["velocity"] == 0.4
    assert calls["merged_recipe"] == "extract-merged"


def test_run_merge_pipeline_requires_subtraction_for_train_difference(monkeypatch):
    monkeypatch.setattr(main.sd_mecha, "model", lambda path: f"recipe::{path}")

    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path

    monkeypatch.setattr(
        "module.utility.load_model",
        lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path),
    )
    monkeypatch.setattr(
        main,
        "get_calculation_strategy",
        lambda strategy_name, replace_with: lambda *args, **kwargs: "diff-node",
    )

    with pytest.raises(main.ConfigError):
        main.run_merge_pipeline(
            {
                "target_model": "models/base.safetensors",
                "models": [
                    {
                        "left": "models/a.safetensors",
                        "right": "models/b.safetensors",
                        "strategy": "addition",
                        "target_strategy": "trainDifference",
                        "key_patterns": ["."],
                    }
                ],
                "output_name": "train_difference_invalid.safetensors",
            },
            default_output_dir=os.getcwd(),
        )


def test_run_merge_pipeline_wires_train_difference_with_target_and_ab_diffs(
    monkeypatch,
):
    calls = {}

    def fake_model(path):
        return f"recipe::{path}"

    def fake_calc_strategy(strategy_name, replace_with):
        assert strategy_name == "subtraction"

        def runner(left_node, right_node, velocity, key_patterns_json):
            return f"diff::{left_node}->{right_node}::{velocity}"

        return runner

    def fake_target_strategy(strategy_name):
        assert strategy_name == "trainDifference"

        def runner(target_node, **kwargs):
            calls["target_kwargs"] = (target_node, kwargs)
            return "train-diff-merged"

        return runner

    monkeypatch.setattr(main.sd_mecha, "model", fake_model)

    class FakeSDKeyWrapper:
        def __init__(self, path):
            self._d = path

    monkeypatch.setattr(
        "module.utility.load_model",
        lambda path, lazy_load=True, use_sdxl_keys=None: FakeSDKeyWrapper(path),
    )
    monkeypatch.setattr(
        main.sd_mecha,
        "merge",
        lambda recipe, output: calls.update({"merged_recipe": recipe, "output": output}),
    )
    monkeypatch.setattr(main, "get_calculation_strategy", fake_calc_strategy)
    monkeypatch.setattr(main, "get_target_calculation_strategy", fake_target_strategy)
    monkeypatch.setattr(main, "run_pre_merge_hooks", lambda config, recipe: recipe)

    main.run_merge_pipeline(
        {
            "target_model": "models/base.safetensors",
            "models": [
                {
                    "left": "models/a.safetensors",
                    "right": "models/b.safetensors",
                    "strategy": "subtraction",
                    "target_strategy": "trainDifference",
                    "velocity": 0.6,
                    "key_patterns": ["."],
                }
            ],
            "output_name": "train_difference_valid.safetensors",
        },
        default_output_dir=os.getcwd(),
    )

    target_node, kwargs = calls["target_kwargs"]
    assert target_node == "recipe::models/base.safetensors"
    assert kwargs["diff_l"] == "diff::recipe::models/base.safetensors->recipe::models/b.safetensors::1.0"
    assert kwargs["diff_r"] == "diff::recipe::models/a.safetensors->recipe::models/b.safetensors::1.0"
    assert kwargs["left"] == "recipe::models/a.safetensors"
    assert kwargs["right"] == "recipe::models/b.safetensors"
    assert kwargs["velocity"] == 0.6
    assert calls["merged_recipe"] == "train-diff-merged"


def test_merge_recipe_uses_castless_output_call_first(monkeypatch):
    calls = []

    def fake_merge(recipe, **kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(main.sd_mecha, "merge", fake_merge)

    result = main._merge_recipe("recipe", output_path="merged.safetensors", dtype="fp16")

    assert result == "ok"
    assert calls == [
        {
            "merge_dtype": "fp16",
            "output_device": None,
            "output_dtype": None,
            "output": "merged.safetensors",
        }
    ]


def test_merge_recipe_falls_back_for_legacy_sd_mecha(monkeypatch):
    calls = []

    def fake_merge(recipe, **kwargs):
        calls.append(kwargs)
        if "merge_dtype" in kwargs:
            raise TypeError("merge() got an unexpected keyword argument 'merge_dtype'")
        if "output_dtype" in kwargs:
            raise TypeError("merge() got an unexpected keyword argument 'output_dtype'")
        return "legacy-ok"

    monkeypatch.setattr(main.sd_mecha, "merge", fake_merge)

    result = main._merge_recipe("recipe", output_path="legacy.safetensors", dtype="fp16")

    assert result == "legacy-ok"
    assert calls == [
        {
            "merge_dtype": "fp16",
            "output_device": None,
            "output_dtype": None,
            "output": "legacy.safetensors",
        },
        {
            "merge_dtype": "fp16",
            "output": "legacy.safetensors",
        },
        {
            "output_dtype": "fp16",
            "output": "legacy.safetensors",
        },
        {
            "output": "legacy.safetensors",
        },
    ]


def test_merge_recipe_retries_on_sd_mecha_node_key_error(monkeypatch):
    calls = []

    def fake_merge(recipe, **kwargs):
        calls.append(kwargs)
        if "merge_dtype" in kwargs or "output_dtype" in kwargs:
            raise KeyError("MergeRecipeNode(method=target_addition, inputs=4 args, 0 kwargs)")
        return "keyerror-recovered"

    monkeypatch.setattr(main.sd_mecha, "merge", fake_merge)

    result = main._merge_recipe("recipe", output_path="recover.safetensors", dtype="fp16")

    assert result == "keyerror-recovered"
    assert calls == [
        {
            "merge_dtype": "fp16",
            "output_device": None,
            "output_dtype": None,
            "output": "recover.safetensors",
        },
        {
            "merge_dtype": "fp16",
            "output": "recover.safetensors",
        },
        {
            "output_dtype": "fp16",
            "output": "recover.safetensors",
        },
        {
            "output": "recover.safetensors",
        },
    ]


def test_merge_recipe_retries_with_in_memory_output_after_streaming_key_error(monkeypatch):
    calls = []

    def fake_merge(recipe, **kwargs):
        calls.append(kwargs)
        if kwargs.get("output") is not None:
            raise KeyError("MergeRecipeNode(method=cast, inputs=3 args, 0 kwargs)")
        if "merge_dtype" in kwargs or "output_dtype" in kwargs:
            raise KeyError("MergeRecipeNode(method=cast, inputs=3 args, 0 kwargs)")
        return {"weight": "tensor"}

    monkeypatch.setattr(main.sd_mecha, "merge", fake_merge)

    result = main._merge_recipe("recipe", output_path="recover.safetensors", dtype="fp16")

    assert result == {"weight": "tensor"}
    assert calls == [
        {
            "merge_dtype": "fp16",
            "output_device": None,
            "output_dtype": None,
            "output": "recover.safetensors",
        },
        {
            "merge_dtype": "fp16",
            "output": "recover.safetensors",
        },
        {
            "output_dtype": "fp16",
            "output": "recover.safetensors",
        },
        {
            "output": "recover.safetensors",
        },
        {
            "merge_dtype": "fp16",
            "output_device": None,
            "output_dtype": None,
            "output": None,
        },
        {
            "merge_dtype": "fp16",
            "output": None,
        },
        {
            "output_dtype": "fp16",
            "output": None,
        },
        {
            "output": None,
        },
    ]



def test_merge_recipe_uses_kwargless_call_when_output_none_still_triggers_cast_key_error(monkeypatch):
    calls = []

    def fake_merge(recipe, **kwargs):
        calls.append(kwargs)
        if kwargs:
            raise KeyError("MergeRecipeNode(method=cast, inputs=3 args, 0 kwargs)")
        return {"weight": "tensor"}

    monkeypatch.setattr(main.sd_mecha, "merge", fake_merge)

    result = main._merge_recipe("recipe", output_path="recover.safetensors", dtype="fp16")

    assert result == {"weight": "tensor"}
    assert calls == [
        {
            "merge_dtype": "fp16",
            "output_device": None,
            "output_dtype": None,
            "output": "recover.safetensors",
        },
        {
            "merge_dtype": "fp16",
            "output": "recover.safetensors",
        },
        {
            "output_dtype": "fp16",
            "output": "recover.safetensors",
        },
        {
            "output": "recover.safetensors",
        },
        {
            "merge_dtype": "fp16",
            "output_device": None,
            "output_dtype": None,
            "output": None,
        },
        {
            "merge_dtype": "fp16",
            "output": None,
        },
        {
            "output_dtype": "fp16",
            "output": None,
        },
        {
            "output": None,
        },
        {},
    ]

def test_run_merge_pipeline_saves_in_memory_fallback_for_non_sharded(monkeypatch):
    calls = {}
    _patch_merge_pipeline_dependencies(monkeypatch, calls)

    saved = {}

    monkeypatch.setattr(
        main,
        "_merge_recipe",
        lambda recipe, *, output_path, dtype: {"weight": "tensor"},
    )
    monkeypatch.setattr(
        "module.utility.save_model",
        lambda model, path: saved.update({"model": model, "path": path}),
    )

    temp_dir = os.path.join(os.getcwd(), ".pytest_tmp_local", str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)

    output_path = main.run_merge_pipeline(
        {
            "target_model": "models/base_model.safetensors",
            "models": [],
            "output_name": "fallback_saved.safetensors",
        },
        default_output_dir=temp_dir,
    )

    assert output_path == os.path.join(temp_dir, "fallback_saved.safetensors")
    assert saved == {
        "model": {"weight": "tensor"},
        "path": output_path,
    }
