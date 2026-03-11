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
