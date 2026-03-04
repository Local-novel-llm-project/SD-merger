import pytest
from pydantic import ValidationError

from module.config_schema import MergeConfig, ModelConfig


def test_valid_model_config():
    config = ModelConfig(left="model_a.safetensors", right="model_b.safetensors", velocity=0.5, strategy="addition")
    assert config.model_A == "model_a.safetensors"
    assert config.strategy == "addition"


def test_invalid_alpha_beta():
    with pytest.raises(ValidationError):
        ModelConfig(left="a.safetensors", right="b.safetensors", alpha=1.5)  # Should be <= 1.0


def test_valid_merge_config():
    raw_dict = {
        "output_dir": "./test_output",
        "models": [{"left": "a.safetensors", "right": "b.safetensors", "velocity": 0.5, "strategy": "mix"}],
        "dtype": "float16",
        "device": "cuda",
    }

    config = MergeConfig(**raw_dict)
    assert config.output_dir == "./test_output"
    assert config.dtype == "float16"
    assert len(config.models) == 1

    # Check that model config was evaluated correctly
    model_cfg = config.models[0]
    # Check if pydantic correctly parsed it as ModelConfig if applicable,
    # Since models is Union[ModelConfig, Dict], and we parsed raw dict,
    # pydantic will try ModelConfig first.
    if isinstance(model_cfg, ModelConfig):
        assert model_cfg.model_A == "a.safetensors"
        assert model_cfg.velocity == 0.5


def test_invalid_dtype():
    raw_dict = {"models": [], "dtype": "invalid_type"}
    with pytest.raises(ValidationError):
        MergeConfig(**raw_dict)
