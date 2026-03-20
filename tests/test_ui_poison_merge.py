import os

from ui.components import poison_merge


def test_build_poison_merge_task_config_includes_preview_generation_settings(monkeypatch):
    monkeypatch.setattr(
        poison_merge,
        "get_model_path",
        lambda model_name: os.path.join("models", model_name),
    )

    config, output_name = poison_merge.build_poison_merge_task_config(
        "base.safetensors",
        ["style_a.safetensors", "style_b.safetensors"],
        2,
        "linear",
        1.0,
        "0.8, 0.4",
        "models/output/poison_merge",
        "test prompt",
        "test negative",
        640,
        768,
        30,
        9.5,
        "dpmpp_2m",
        "karras",
        123,
    )

    poison_config = config["poison_merge"]

    assert output_name == "poison_step_2_alpha_0.40.safetensors"
    assert config["output_name"] == output_name
    assert poison_config["base_model"] == os.path.join("models", "base.safetensors")
    assert poison_config["lora_model"] == os.path.join("models", "style_a.safetensors")
    assert poison_config["lora_models"] == [
        os.path.join("models", "style_a.safetensors"),
        os.path.join("models", "style_b.safetensors"),
    ]
    assert poison_config["prompt"] == "test prompt"
    assert poison_config["negative_prompt"] == "test negative"
    assert poison_config["width"] == 640
    assert poison_config["height"] == 768
    assert poison_config["steps"] == 30
    assert poison_config["cfg"] == 9.5
    assert poison_config["sampler_name"] == "dpmpp_2m"
    assert poison_config["scheduler"] == "karras"
    assert poison_config["seed"] == 123
