import os

import ui.services.model_service as model_service
import ui.utils as utils


def test_list_models_scans_supported_extensions(monkeypatch):
    monkeypatch.setattr(model_service, "get_models_dir", lambda: "C:\\models")
    monkeypatch.setattr(model_service.os.path, "exists", lambda path: path == "C:\\models")

    walk_rows = [
        ("C:\\models", ["nested"], ["a.safetensors", "b.txt", "c.ckpt"]),
        ("C:\\models\\nested", [], ["d.pt", "e.bin"]),
    ]
    monkeypatch.setattr(model_service.os, "walk", lambda *args, **kwargs: iter(walk_rows))

    models = model_service.list_models()

    assert models == [
        "a.safetensors",
        "c.ckpt",
        os.path.join("nested", "d.pt"),
        os.path.join("nested", "e.bin"),
    ]


def test_resolve_model_path_joins_models_dir(monkeypatch):
    monkeypatch.setattr(model_service, "get_models_dir", lambda: "/repo/models")

    assert model_service.resolve_model_path("foo/bar.safetensors") == "/repo/models/foo/bar.safetensors"


def test_ui_utils_reexports_model_helpers():
    assert utils.get_models_dir is model_service.get_models_dir
    assert utils.get_model_list is model_service.list_models
    assert utils.get_model_path is model_service.resolve_model_path
