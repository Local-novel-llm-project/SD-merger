import os
import sys
import types

from extensions import lora_ops


def test_run_lora_operations_apply_sets_skip_merge_output(monkeypatch):
    called = []

    def fake_run_merge(op):
        called.append(op)
        return op["output"]

    monkeypatch.setattr(lora_ops, "_run_merge_lora", fake_run_merge)

    config = {
        "lora_ops": {
            "stop_after_lora_ops": True,
            "operations": [
                {
                    "type": "apply",
                    "sd_model": "base_model.safetensors",
                    "models": ["style_lora.safetensors"],
                    "ratios": [0.75],
                    "output": "merged_checkpoint.safetensors",
                }
            ],
        }
    }

    result = lora_ops.run_lora_operations(config)

    assert called == [config["lora_ops"]["operations"][0]]
    assert result["_skip_merge"] is True
    assert result["_skip_merge_output"] == "merged_checkpoint.safetensors"


def test_run_merge_lora_forwards_sd_model_to_selected_runner(monkeypatch):
    captured = {}

    def fake_runner(args):
        captured["args"] = args

    def fake_select_merge_runner(is_sdxl):
        captured["is_sdxl"] = is_sdxl
        return fake_runner

    monkeypatch.setattr(lora_ops, "_select_merge_runner", fake_select_merge_runner)

    output_path = lora_ops._run_merge_lora(
        {
            "models": ["style_lora.safetensors"],
            "ratios": [1.0],
            "sd_model": "base_model.safetensors",
            "output": "merged_checkpoint.safetensors",
            "precision": "fp16",
            "save_precision": "bf16",
            "sdxl": True,
            "v2": False,
        }
    )

    assert captured["is_sdxl"] is True
    assert captured["args"].sd_model == "base_model.safetensors"
    assert captured["args"].models == ["style_lora.safetensors"]
    assert captured["args"].ratios == [1.0]
    assert captured["args"].precision == "fp16"
    assert captured["args"].save_precision == "bf16"
    assert output_path == "merged_checkpoint.safetensors"


def test_run_merge_lora_supports_compact_model_ratio_syntax(monkeypatch):
    captured = {}

    def fake_runner(args):
        captured["args"] = args

    monkeypatch.setattr(lora_ops, "_select_merge_runner", lambda is_sdxl: fake_runner)

    output_path = lora_ops._run_merge_lora(
        {
            "models": "style_a.safetensors:0.4, style_b.safetensors:0.9",
            "output": "merged_lora.safetensors",
        }
    )

    assert captured["args"].models == [
        "style_a.safetensors",
        "style_b.safetensors",
    ]
    assert captured["args"].ratios == [0.4, 0.9]
    assert output_path == "merged_lora.safetensors"


def test_ensure_kohya_import_aliases_registers_vendor_namespace_packages():
    original_modules = {}
    target_names = ["scripts", "scripts.kohyas", "library"]

    try:
        for name in target_names:
            original_modules[name] = sys.modules.pop(name, None)

        kohyas_dir = lora_ops._ensure_kohya_import_aliases()

        assert sys.modules["scripts"].__path__ == [os.path.dirname(kohyas_dir)]
        assert sys.modules["scripts.kohyas"].__path__ == [kohyas_dir]
        assert sys.modules["library"].__path__ == [kohyas_dir]
    finally:
        for name in target_names:
            sys.modules.pop(name, None)
        for name, module in original_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_load_kohya_symbol_imports_local_vendor_module(monkeypatch):
    fake_symbol_module = types.SimpleNamespace(svd="callable")
    fake_train_util = types.SimpleNamespace(
        load_metadata_from_safetensors=lambda _: {},
        build_minimum_network_metadata=lambda *args: {},
        SS_METADATA_KEY_V2="ss_v2",
        SS_METADATA_KEY_BASE_MODEL_VERSION="ss_base_model_version",
    )
    fake_sai_model_spec = types.SimpleNamespace(
        load_metadata_from_safetensors=lambda _: {},
    )
    calls = []

    def fake_import_module(name, package=None):
        calls.append((name, package))
        if name in {
            "scripts.kohyas.train_util",
            "library.train_util",
            "extensions.lora_ops.kohyas.train_util",
        }:
            return fake_train_util
        if name in {
            "scripts.kohyas.sai_model_spec",
            "library.sai_model_spec",
            "extensions.lora_ops.kohyas.sai_model_spec",
        }:
            return fake_sai_model_spec
        return fake_symbol_module

    monkeypatch.setattr(lora_ops.importlib, "import_module", fake_import_module)

    symbol = lora_ops._load_kohya_symbol("extract_lora_from_models", "svd")

    assert symbol == "callable"
    assert calls[-1] == (
        ".kohyas.extract_lora_from_models",
        "extensions.lora_ops",
    )


def test_ensure_kohya_train_util_compat_adds_missing_metadata_helpers():
    def fake_load_metadata(path):
        return {"loaded_from": path}

    train_util_module = types.SimpleNamespace()
    sai_model_spec_module = types.SimpleNamespace(
        load_metadata_from_safetensors=fake_load_metadata
    )

    lora_ops._ensure_kohya_train_util_compat(
        train_util_module,
        sai_model_spec_module,
    )

    assert train_util_module.load_metadata_from_safetensors is fake_load_metadata
    assert train_util_module.SS_METADATA_KEY_V2 == "ss_v2"
    assert train_util_module.SS_METADATA_KEY_BASE_MODEL_VERSION == (
        "ss_base_model_version"
    )
    assert train_util_module.build_minimum_network_metadata(
        True,
        "sdxl",
        "networks.lora",
        "128",
        "128",
        {"conv_dim": 16},
    ) == {
        "ss_v2": "True",
        "ss_base_model_version": "sdxl",
        "ss_network_module": "networks.lora",
        "ss_network_dim": "128",
        "ss_network_alpha": "128",
        "ss_network_args": '{"conv_dim": 16}',
    }
