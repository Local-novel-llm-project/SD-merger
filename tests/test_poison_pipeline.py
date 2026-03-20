import sys
import types
from pathlib import Path

from module.pipeline import poison


def test_normalize_lora_models_supports_legacy_single_value():
    poison_config = {"lora_model": "loras/style.safetensors"}

    assert normalize_lora_models(poison_config) == ["loras/style.safetensors"]


def test_normalize_lora_models_prefers_multi_select_list():
    poison_config = {
        "lora_model": "loras/legacy.safetensors",
        "lora_models": [
            "loras/style_a.safetensors",
            "",
            "  loras/style_b.safetensors  ",
        ],
    }

    assert normalize_lora_models(poison_config) == [
        "loras/style_a.safetensors",
        "loras/style_b.safetensors",
    ]


def test_build_iteration_plan_chains_multiple_loras():
    plan = build_iteration_plan(
        current_base="models/base.safetensors",
        lora_models=["loras/a.safetensors", "loras/b.safetensors"],
        output_dir="models/output/poison_merge",
        iteration=2,
        alpha=0.75,
    )

    assert len(plan) == 2
    assert plan[0]["left"] == "models/base.safetensors"
    assert plan[0]["right"] == "loras/a.safetensors"
    assert plan[0]["output_name"] == "poison_step_2_lora_1_alpha_0.75.safetensors"
    assert plan[1]["left"] == plan[0]["output_path"]
    assert plan[1]["right"] == "loras/b.safetensors"
    assert plan[1]["output_name"] == "poison_step_2_alpha_0.75.safetensors"


def test_resolve_lora_merge_precision_maps_supported_dtypes():
    assert poison.resolve_lora_merge_precision("float16") == "fp16"
    assert poison.resolve_lora_merge_precision("float32") == "float"
    assert poison.resolve_lora_merge_precision("bfloat16") == "bf16"
    assert poison.resolve_lora_merge_precision("unknown") == "float"


def test_build_poison_lora_apply_operation_uses_lora_ops_apply_shape():
    operation = poison.build_poison_lora_apply_operation(
        "models/base.safetensors",
        "loras/style.safetensors",
        0.6,
        "output/merged.safetensors",
        sdxl=True,
        precision="fp16",
        save_precision="bf16",
        v2=True,
        no_metadata=True,
    )

    assert operation == {
        "type": "apply",
        "sd_model": "models/base.safetensors",
        "models": ["loras/style.safetensors"],
        "ratios": [0.6],
        "output": "output/merged.safetensors",
        "sdxl": True,
        "precision": "fp16",
        "save_precision": "bf16",
        "v2": True,
        "no_metadata": True,
    }


def test_run_poison_merge_applies_lora_stages_via_lora_ops(monkeypatch, tmp_path):
    output_dir = tmp_path / "poison_out"
    apply_calls = []

    monkeypatch.setattr(poison, "infer_model_is_sdxl", lambda model_path: True)

    fake_generation = types.ModuleType("module.generation")
    fake_generation.generate_first_image = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "module.generation", fake_generation)

    fake_pil = types.ModuleType("PIL")

    class FakeImageModule:
        @staticmethod
        def new(*args, **kwargs):
            class FakeImage:
                def paste(self, *a, **k):
                    return None

                def save(self, *a, **k):
                    return None

            return FakeImage()

    class FakeImageDrawModule:
        @staticmethod
        def Draw(image):
            class FakeDraw:
                def rectangle(self, *a, **k):
                    return None

                def text(self, *a, **k):
                    return None

            return FakeDraw()

    fake_pil.Image = FakeImageModule
    fake_pil.ImageDraw = FakeImageDrawModule
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    def fake_apply_lora_stage(
        base_model,
        lora_model,
        alpha,
        output_path,
        *,
        sdxl,
        precision,
        save_precision,
        v2=False,
        no_metadata=False,
    ):
        apply_calls.append(
            {
                "base_model": base_model,
                "lora_model": lora_model,
                "alpha": alpha,
                "output_path": output_path,
                "sdxl": sdxl,
                "precision": precision,
                "save_precision": save_precision,
                "v2": v2,
                "no_metadata": no_metadata,
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("merged", encoding="utf-8")
        return output_path

    monkeypatch.setattr(poison, "apply_lora_stage", fake_apply_lora_stage)

    config = {
        "dtype": "float16",
        "poison_merge": {
            "base_model": "models/base.safetensors",
            "lora_models": ["loras/style.safetensors"],
            "iterations": 1,
            "initial_alpha": 0.75,
            "output_dir": str(output_dir),
            "prompt": "test prompt",
            "negative_prompt": "test negative",
            "seed": 123,
        },
    }

    result = poison.run_poison_merge(config, task_name="Poison Merge")

    expected_output = output_dir / "poison_step_1_alpha_0.75.safetensors"
    assert result == str(expected_output)
    assert apply_calls == [
        {
            "base_model": "models/base.safetensors",
            "lora_model": "loras/style.safetensors",
            "alpha": 0.75,
            "output_path": str(expected_output),
            "sdxl": True,
            "precision": "fp16",
            "save_precision": "fp16",
            "v2": False,
            "no_metadata": False,
        }
    ]


normalize_lora_models = poison.normalize_lora_models
build_iteration_plan = poison.build_iteration_plan
