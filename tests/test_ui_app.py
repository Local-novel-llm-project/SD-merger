import argparse

from ui import app


def test_resolve_launch_options_defaults_to_localhost():
    options = app.resolve_launch_options(
        argparse.Namespace(listen=False, port=7860)
    )

    assert options["server_name"] == app.DEFAULT_HOST
    assert options["server_port"] == app.DEFAULT_PORT
    assert options["share"] is False
    assert "gradio-container" in options["head"]


def test_resolve_launch_options_uses_listen_and_custom_port():
    options = app.resolve_launch_options(
        argparse.Namespace(listen=True, port=9000)
    )

    assert options["server_name"] == "0.0.0.0"
    assert options["server_port"] == 9000


def test_create_arg_parser_supports_listen_and_port():
    parser = app.create_arg_parser()

    args = parser.parse_args(["--listen", "--port", "9001"])

    assert args.listen is True
    assert args.port == 9001


def test_build_merge_task_config_omits_left_right_velocity_by_default(monkeypatch):
    monkeypatch.setattr(app, "get_model_path", lambda name: f"/models/{name}")
    monkeypatch.setattr(app, "generate_filename", lambda a, b: f"{a}_{b}.safetensors")

    config, output_name = app._build_merge_task_config(
        a="ModelA",
        b="ModelB",
        c="選択しない",
        strat="mix",
        t_strat="mix",
        vel=0.5,
        left_right_vel="",
        use_adv=False,
        mbw="",
        vae="",
        out="custom.safetensors",
        lazy_load_opt=True,
    )

    assert config["target_model"] == ""
    assert config["lazy_load"] is True
    assert config["models"][0]["velocity"] == 0.5
    assert "left_right_velocity" not in config["models"][0]
    assert output_name == app._create_default_merge_output_name("ModelA", "ModelB")


def test_build_merge_task_config_includes_optional_left_right_velocity(monkeypatch):
    monkeypatch.setattr(app, "get_model_path", lambda name: f"/models/{name}")
    monkeypatch.setattr(app, "generate_filename", lambda a, b: f"{a}_{b}.safetensors")

    config, output_name = app._build_merge_task_config(
        a="ModelA",
        b="ModelB",
        c="ModelC",
        strat="subtraction",
        t_strat="addition",
        vel=0.25,
        left_right_vel="0.75",
        use_adv=True,
        mbw="",
        vae="VAE",
        out="merged.safetensors",
        lazy_load_opt=False,
    )

    assert config["target_model"] == "/models/ModelC"
    assert config["lazy_load"] is False
    assert config["models"][0]["left_right_velocity"] == 0.75
    assert config["bake_in_vae"] == "/models/VAE"
    assert config["output_name"] == "merged.safetensors"
    assert output_name == "merged.safetensors"
