from ui.components import multi_merge


def test_parse_multi_merge_command_supports_left_right_velocity_aliases():
    operations = multi_merge.parse_multi_merge_command(
        "O=merged.safetensors, S=mix, base_alpha=0.5, LRV=0.75, Model_A=model_a.safetensors, Model_B=model_b.safetensors"
    )

    assert operations == [
        {
            "strategy": "mix",
            "output_name": "merged.safetensors",
            "velocity": 0.5,
            "left_right_velocity": 0.75,
            "left": "model_a.safetensors",
            "right": "model_b.safetensors",
        }
    ]


def test_parse_multi_merge_command_supports_long_left_right_velocity_names():
    operations = multi_merge.parse_multi_merge_command(
        "O=merged.safetensors, Strategy=addition, Left_Right_Velocity=0.6, Model_A=model_a.safetensors, Model_B=model_b.safetensors"
    )

    assert operations == [
        {
            "strategy": "addition",
            "output_name": "merged.safetensors",
            "left_right_velocity": 0.6,
            "left": "model_a.safetensors",
            "right": "model_b.safetensors",
        }
    ]
