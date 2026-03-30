from pathlib import Path

from PIL import Image

from module.services.ab_test import build_ab_test_configs, run_ab_test
from module.services.generation import ArtifactDTO, GenerationResult


def test_build_ab_test_configs_swaps_model_order(monkeypatch):
    monkeypatch.setattr(
        "module.services.ab_test.build_basic_merge_config",
        lambda model_a, model_b, *args, **kwargs: (
            {"models": [{"left": model_a, "right": model_b}]},
            f"{model_a}_{model_b}.safetensors",
        ),
    )

    configs = build_ab_test_configs(
        model_a="A",
        model_b="B",
        model_c=None,
        strategy="mix",
        target_strategy="mix",
        velocity=0.5,
        left_right_velocity=None,
        use_advanced_options=False,
        mbw=None,
        bake_in_vae=None,
        lazy_load=True,
    )

    assert configs["A->B"][0]["models"][0] == {"left": "A", "right": "B"}
    assert configs["B->A"][0]["models"][0] == {"left": "B", "right": "A"}


def test_run_ab_test_returns_generation_and_plot_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "module.services.ab_test.build_ab_test_configs",
        lambda **kwargs: {
            "A->B": ({"models": [{"left": "A", "right": "B"}]}, "ab_a_b.safetensors"),
            "B->A": ({"models": [{"left": "B", "right": "A"}]}, "ab_b_a.safetensors"),
        },
    )

    def fake_generation_runner(request):
        image = Image.new("RGB", (4, 4), color="white")
        image_path = Path(request.output_dir) / f"{request.output_prefix}.png"
        image.save(image_path)
        artifact = ArtifactDTO(kind="image", path=str(image_path), label=request.artifact_label_prefix)
        return GenerationResult(seed=7, artifacts=(artifact,), preview=artifact, images=(image,))

    result = run_ab_test(
        model_a="A",
        model_b="B",
        prompt="prompt",
        output_dir=str(tmp_path),
        merge_runner=lambda config, output_dir: str(Path(output_dir) / "merged.safetensors"),
        generation_runner=fake_generation_runner,
        clip_score_fn=lambda images, prompt: [0.2],
        radar_chart_fn=lambda metrics, title: Image.new("RGB", (8, 8), color="black"),
    )

    assert len(result.runs) == 2
    assert set(result.metrics) == {"A->B", "B->A"}
    assert len(result.artifacts) == 3
    assert result.artifacts[-1].kind == "plot"
