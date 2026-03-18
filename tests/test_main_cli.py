import main


def test_parse_cli_args_supports_legacy_merge_mode():
    args = main._parse_cli_args(["-c", "sample.yaml", "-o", "merged"])

    assert args.command == "merge"
    assert args.config == "sample.yaml"
    assert args.output == "merged"


def test_build_tune_config_from_cli_args():
    vectors = ",".join(["1.0"] * 19)
    args = main._parse_cli_args(
        [
            "tune",
            "--model",
            "models/base.safetensors",
            "--mode",
            "Real Value",
            "--semantic-focus",
            "1.2",
            "--out-shadows-depth",
            "1.15",
            "--vectors-override",
            vectors,
            "--output-name",
            "arthemy_cli.safetensors",
        ]
    )

    config = main._build_tune_config_from_args(args)

    assert args.command == "tune"
    assert config["target_model"] == "models/base.safetensors"
    assert config["output_name"] == "arthemy_cli.safetensors"
    assert config["arthemy_tuner"]["mode"] == "Real Value"
    assert config["arthemy_tuner"]["clip"]["semantic_focus"] == 1.2
    assert config["arthemy_tuner"]["unet"]["OUT_Shadows_Depth"] == 1.15
    assert config["arthemy_tuner"]["unet"]["vectors_override"] == [1.0] * 19


def test_parse_cli_args_supports_ui_subcommand():
    args = main._parse_cli_args(["ui", "--port", "7861", "--share"])

    assert args.command == "ui"
    assert args.port == 7861
    assert args.share is True


def test_parse_cli_args_supports_global_debug_before_subcommand():
    args = main._parse_cli_args(["--debug", "tune", "--model", "models/base.safetensors"])

    assert args.command == "tune"
    assert args.debug is True
    assert args.model == "models/base.safetensors"
