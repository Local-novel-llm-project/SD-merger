from module.pipeline.poison import (
    build_iteration_plan,
    normalize_lora_models,
)


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
