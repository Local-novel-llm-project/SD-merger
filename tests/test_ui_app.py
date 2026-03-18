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
