import os

import module.services.merge as merge_services
from module.services.merge import model_catalog
import ui.services.model_service as model_service
import ui.utils as utils


def test_list_models_scans_supported_extensions(monkeypatch):
    monkeypatch.setattr(model_catalog, "get_models_dir", lambda: "C:\\models")
    monkeypatch.setattr(
        model_catalog.os.path,
        "exists",
        lambda path: path == "C:\\models",
    )

    walk_rows = [
        ("C:\\models", ["nested"], ["a.safetensors", "b.txt", "c.ckpt"]),
        ("C:\\models\\nested", [], ["d.pt", "e.bin"]),
    ]
    monkeypatch.setattr(
        model_catalog.os,
        "walk",
        lambda *args, **kwargs: iter(walk_rows),
    )

    models = model_catalog.list_models()

    assert models == [
        "a.safetensors",
        "c.ckpt",
        os.path.join("nested", "d.pt"),
        os.path.join("nested", "e.bin"),
    ]


def test_resolve_model_path_joins_models_dir(monkeypatch):
    monkeypatch.setattr(model_catalog, "get_models_dir", lambda: "/repo/models")

    assert model_catalog.resolve_model_path("foo/bar.safetensors") == "/repo/models/foo/bar.safetensors"


def test_ui_utils_reexports_model_helpers():
    assert model_service.get_models_dir is merge_services.get_models_dir
    assert model_service.list_models is merge_services.list_models
    assert model_service.resolve_model_path is merge_services.resolve_model_path
    assert utils.get_models_dir is merge_services.get_models_dir
    assert utils.get_model_list is merge_services.list_models
    assert utils.get_model_path is merge_services.resolve_model_path
