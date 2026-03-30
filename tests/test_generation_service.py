from pathlib import Path

from PIL import Image

from module.services.generation import (
    GenerationRequest,
    build_history_image_update,
    create_plot_artifact,
    generate_and_collect_artifacts,
)


def test_generate_and_collect_artifacts_returns_saved_dtos(tmp_path):
    image = Image.new("RGB", (4, 4), color="white")

    result = generate_and_collect_artifacts(
        GenerationRequest(
            model_path="model.safetensors",
            output_dir=str(tmp_path),
            output_prefix="sample",
            prompt="prompt",
            seed=10,
        ),
        generate_fn=lambda **kwargs: [image],
    )

    assert result.seed == 10
    assert len(result.artifacts) == 1
    assert result.preview == result.artifacts[0]
    assert Path(result.artifacts[0].path).exists()
    assert result.artifacts[0].kind == "image"


def test_build_history_image_update_uses_saved_artifacts():
    update_dict = build_history_image_update(
        generate_and_collect_artifacts(
            GenerationRequest(
                model_path="model.safetensors",
                output_dir=".",
                output_prefix="sample",
                prompt="prompt",
                seed=3,
            ),
            generate_fn=lambda **kwargs: [],
        )
    )

    assert update_dict == {}


def test_create_plot_artifact_saves_plot_image(tmp_path):
    image = Image.new("RGB", (8, 8), color="black")

    artifact = create_plot_artifact(
        image,
        output_dir=str(tmp_path),
        file_name="plot.png",
        label="Plot",
        metadata={"kind": "radar"},
    )

    assert artifact.kind == "plot"
    assert artifact.label == "Plot"
    assert artifact.metadata["kind"] == "radar"
    assert Path(artifact.path).exists()
