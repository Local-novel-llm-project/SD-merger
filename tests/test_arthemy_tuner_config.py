import pytest

from module.arthemy_tuner_config import (
    build_arthemy_tune_job_config,
    normalize_vectors_override,
)


def test_normalize_vectors_override_parses_comma_separated_values():
    parsed = normalize_vectors_override(",".join(["1.0"] * 19))

    assert parsed == [1.0] * 19


def test_build_arthemy_tune_job_config_includes_defaults_and_override():
    config = build_arthemy_tune_job_config(
        target_model="models/base.safetensors",
        mode="Real Value",
        clip_overrides={"semantic_focus": 1.15},
        unet_overrides={
            "OUT_Shadows_Depth": 1.2,
            "vectors_override": ",".join(["1.0"] * 19),
        },
        output_name="arthemy_tuned.safetensors",
    )

    assert config["target_model"] == "models/base.safetensors"
    assert config["models"] == []
    assert config["output_name"] == "arthemy_tuned.safetensors"
    assert config["arthemy_tuner"]["mode"] == "Real Value"
    assert config["arthemy_tuner"]["clip"]["semantic_focus"] == 1.15
    assert config["arthemy_tuner"]["unet"]["OUT_Shadows_Depth"] == 1.2
    assert config["arthemy_tuner"]["unet"]["vectors_override"] == [1.0] * 19


def test_build_arthemy_tune_job_config_rejects_invalid_vector_count():
    with pytest.raises(ValueError):
        build_arthemy_tune_job_config(
            target_model="models/base.safetensors",
            unet_overrides={"vectors_override": "1.0,1.1"},
        )
