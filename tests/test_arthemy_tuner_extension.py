import importlib


arthemy_tuner = importlib.import_module("extensions.arthemy_tuner")


def test_build_unet_data_prefers_vectors_override():
    unet_data = arthemy_tuner._build_unet_data(
        {
            "base_scale": 0.95,
            "config_dict": {"IN_Layout_Geometry": 0.25},
            "vectors_override": [1.25] + [1.0] * 18,
        },
        "Real Value",
    )

    assert (
        arthemy_tuner._get_unet_target_weight(
            "model.diffusion_model.input_blocks.0.0.weight",
            unet_data,
        )
        == 1.25
    )
    assert unet_data[1] == 0.95


def test_build_unet_data_falls_back_to_grouped_values_when_override_is_invalid():
    unet_data = arthemy_tuner._build_unet_data(
        {
            "base_scale": 1.0,
            "config_dict": {"OUT_Shadows_Depth": 1.3},
            "vectors_override": [1.0, 1.1],
        },
        "Real Value",
    )

    assert (
        arthemy_tuner._get_unet_target_weight(
            "model.diffusion_model.output_blocks.5.0.weight",
            unet_data,
        )
        == 1.3
    )
