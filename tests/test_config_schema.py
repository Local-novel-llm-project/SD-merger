import pytest
from pydantic import ValidationError

from module.config_schema import GenerationConfig, MergeConfig, ModelConfig


def test_valid_model_config():
    config = ModelConfig(
        left="model_a.safetensors",
        right="model_b.safetensors",
        velocity=0.5,
        strategy="addition",
    )
    assert config.left == "model_a.safetensors"
    assert config.model_A == "model_a.safetensors"
    assert config.alpha == 0.5
    assert config.strategy == "addition"


def test_legacy_aliases_are_supported():
    config = ModelConfig(model_a="a.safetensors", model_b="b.safetensors", alpha=0.25)
    assert config.left == "a.safetensors"
    assert config.right == "b.safetensors"
    assert config.velocity == 0.25


def test_invalid_velocity_range():
    with pytest.raises(ValidationError):
        ModelConfig(left="a.safetensors", right="b.safetensors", velocity=1.5)


def test_valid_merge_config():
    raw_dict = {
        "output_dir": "./test_output",
        "models": [
            {
                "left": "a.safetensors",
                "right": "b.safetensors",
                "velocity": 0.5,
                "strategy": "mix",
            }
        ],
        "dtype": "float16",
        "device": "cuda",
    }

    config = MergeConfig(**raw_dict)
    assert config.output_dir == "./test_output"
    assert config.dtype == "float16"
    assert len(config.models) == 1

    model_cfg = config.models[0]
    assert isinstance(model_cfg, ModelConfig)
    assert model_cfg.left == "a.safetensors"
    assert model_cfg.right == "b.safetensors"
    assert model_cfg.velocity == 0.5


def test_invalid_dtype():
    raw_dict = {"models": [], "dtype": "invalid_type"}
    with pytest.raises(ValidationError):
        MergeConfig(**raw_dict)


def test_generation_config_uses_supported_sampler_default():
    config = GenerationConfig()
    assert config.sampler == "euler_ancestral"
