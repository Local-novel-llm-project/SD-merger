import sys
import types
import shutil
import uuid
from pathlib import Path
import random

from module.pipeline import poison


def _make_runtime_dir(name: str) -> Path:
    root = Path(__file__).parent / ".runtime"
    root.mkdir(exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir()
    return path


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


def test_build_iteration_plan_batches_multiple_loras_into_single_output():
    plan = build_iteration_plan(
        current_base="models/base.safetensors",
        lora_models=["loras/a.safetensors", "loras/b.safetensors"],
        output_dir="models/output/poison_merge",
        iteration=2,
        alpha=0.75,
    )

    assert len(plan) == 1
    assert plan[0]["left"] == "models/base.safetensors"
    assert plan[0]["models"] == ["loras/a.safetensors", "loras/b.safetensors"]
    assert plan[0]["ratios"] == [0.75, 0.75]
    assert plan[0]["output_name"] == "poison_step_2_alpha_0.75.safetensors"


def test_resolve_lora_merge_precision_maps_supported_dtypes():
    assert poison.resolve_lora_merge_precision("float16") == "fp16"
    assert poison.resolve_lora_merge_precision("float32") == "float"
    assert poison.resolve_lora_merge_precision("bfloat16") == "bf16"
    assert poison.resolve_lora_merge_precision("unknown") == "float"


def test_resolve_poison_output_name_uses_last_override_alpha():
    assert poison.resolve_poison_output_name(
        {
            "iterations": 4,
            "initial_alpha": 1.0,
            "decay_type": "linear",
            "alpha_overrides": "0.8, 0.55, 0.2",
        }
    ) == "poison_step_3_alpha_0.20.safetensors"


def test_list_poison_preview_entries_returns_grid_then_sorted_steps():
    runtime_dir = _make_runtime_dir("poison_preview_entries")

    try:
        (runtime_dir / poison.POISON_GRID_IMAGE_NAME).write_text("grid", encoding="utf-8")
        (runtime_dir / "poison_step_10_alpha_0.10_preview.png").write_text(
            "step10",
            encoding="utf-8",
        )
        (runtime_dir / "poison_step_2_alpha_0.50_preview.png").write_text(
            "step2",
            encoding="utf-8",
        )

        entries = poison.list_poison_preview_entries(str(runtime_dir))

        assert entries == [
            {
                "label": "Grid",
                "path": str(runtime_dir / poison.POISON_GRID_IMAGE_NAME),
            },
            {
                "label": "Step 2 (alpha 0.50)",
                "path": str(runtime_dir / "poison_step_2_alpha_0.50_preview.png"),
            },
            {
                "label": "Step 10 (alpha 0.10)",
                "path": str(runtime_dir / "poison_step_10_alpha_0.10_preview.png"),
            },
        ]
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_build_poison_lora_apply_operation_uses_lora_ops_apply_shape():
    operation = poison.build_poison_lora_apply_operation(
        "models/base.safetensors",
        ["loras/style_a.safetensors", "loras/style_b.safetensors"],
        [0.6, 0.6],
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
        "models": ["loras/style_a.safetensors", "loras/style_b.safetensors"],
        "ratios": [0.6, 0.6],
        "output": "output/merged.safetensors",
        "sdxl": True,
        "precision": "fp16",
        "save_precision": "bf16",
        "v2": True,
        "no_metadata": True,
    }


def test_run_poison_merge_applies_all_loras_in_single_stage_via_lora_ops(monkeypatch):
    runtime_dir = _make_runtime_dir("poison_lora_ops")
    output_dir = runtime_dir / "poison_out"
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
        lora_models,
        ratios,
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
                "lora_models": lora_models,
                "ratios": ratios,
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
            "lora_models": [
                "loras/style_a.safetensors",
                "loras/style_b.safetensors",
            ],
            "iterations": 1,
            "initial_alpha": 0.75,
            "output_dir": str(output_dir),
            "prompt": "test prompt",
            "negative_prompt": "test negative",
            "seed": 123,
        },
    }

    try:
        result = poison.run_poison_merge(config, task_name="Poison Merge")

        expected_output = output_dir / "poison_step_1_alpha_0.75.safetensors"
        assert result == str(expected_output)
        assert apply_calls == [
            {
                "base_model": "models/base.safetensors",
                "lora_models": [
                    "loras/style_a.safetensors",
                    "loras/style_b.safetensors",
                ],
                "ratios": [0.75, 0.75],
                "output_path": str(expected_output),
                "sdxl": True,
                "precision": "fp16",
                "save_precision": "fp16",
                "v2": False,
                "no_metadata": False,
            }
        ]
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_run_poison_merge_uses_preview_settings_and_reuses_seed(monkeypatch):
    runtime_dir = _make_runtime_dir("poison_preview_settings")
    output_dir = runtime_dir / "poison_out"
    generation_calls = []
    fake_images = []

    monkeypatch.setattr(poison, "infer_model_is_sdxl", lambda model_path: False)
    monkeypatch.setattr(random, "randint", lambda _start, _end: 4242)

    class FakeGeneratedImage:
        def __init__(self):
            self.saved_paths = []

        def save(self, path):
            self.saved_paths.append(path)
            Path(path).write_text("preview", encoding="utf-8")

    fake_generation = types.ModuleType("module.generation")

    def fake_generate_first_image(**kwargs):
        generation_calls.append(kwargs)
        image = FakeGeneratedImage()
        fake_images.append(image)
        return image

    fake_generation.generate_first_image = fake_generate_first_image
    monkeypatch.setitem(sys.modules, "module.generation", fake_generation)

    fake_pil = types.ModuleType("PIL")

    class FakeGridImage:
        def paste(self, *args, **kwargs):
            return None

        def save(self, path):
            Path(path).write_text("grid", encoding="utf-8")

    class FakeImageModule:
        @staticmethod
        def new(*args, **kwargs):
            return FakeGridImage()

    class FakeImageDrawModule:
        @staticmethod
        def Draw(image):
            class FakeDraw:
                def rectangle(self, *args, **kwargs):
                    return None

                def text(self, *args, **kwargs):
                    return None

            return FakeDraw()

    fake_pil.Image = FakeImageModule
    fake_pil.ImageDraw = FakeImageDrawModule
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    def fake_apply_lora_stage(
        base_model,
        lora_models,
        ratios,
        output_path,
        *,
        sdxl,
        precision,
        save_precision,
        v2=False,
        no_metadata=False,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("merged", encoding="utf-8")
        return output_path

    monkeypatch.setattr(poison, "apply_lora_stage", fake_apply_lora_stage)

    config = {
        "dtype": "float32",
        "poison_merge": {
            "base_model": "models/base.safetensors",
            "lora_models": ["loras/style_a.safetensors"],
            "iterations": 2,
            "initial_alpha": 0.6,
            "alpha_overrides": "0.6, 0.3",
            "output_dir": str(output_dir),
            "prompt": "test prompt",
            "negative_prompt": "test negative",
            "width": 640,
            "height": 768,
            "steps": 30,
            "cfg": 9.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "seed": -1,
        },
    }

    try:
        result = poison.run_poison_merge(config, task_name="Poison Merge")

        assert result == str(output_dir / "poison_step_2_alpha_0.30.safetensors")
        assert [call["seed"] for call in generation_calls] == [4242, 4242]
        assert [call["width"] for call in generation_calls] == [640, 640]
        assert [call["height"] for call in generation_calls] == [768, 768]
        assert [call["steps"] for call in generation_calls] == [30, 30]
        assert [call["cfg"] for call in generation_calls] == [9.5, 9.5]
        assert [call["sampler_name"] for call in generation_calls] == [
            "dpmpp_2m",
            "dpmpp_2m",
        ]
        assert [call["scheduler"] for call in generation_calls] == [
            "karras",
            "karras",
        ]
        assert fake_images[0].saved_paths == [
            str(output_dir / "poison_step_1_alpha_0.60_preview.png")
        ]
        assert fake_images[1].saved_paths == [
            str(output_dir / "poison_step_2_alpha_0.30_preview.png")
        ]
        assert (output_dir / poison.POISON_GRID_IMAGE_NAME).exists()
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


normalize_lora_models = poison.normalize_lora_models
build_iteration_plan = poison.build_iteration_plan
