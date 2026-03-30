from __future__ import annotations

import os


VALID_MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".bin")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def get_models_dir() -> str:
    return os.path.join(PROJECT_ROOT, "models")


def list_models() -> list[str]:
    models_dir = get_models_dir()
    if not os.path.exists(models_dir):
        return []

    model_files: list[str] = []
    for root, _, files in os.walk(models_dir, followlinks=True):
        for file_name in files:
            if file_name.endswith(VALID_MODEL_EXTENSIONS):
                rel_path = os.path.relpath(os.path.join(root, file_name), models_dir)
                model_files.append(rel_path)

    return sorted(model_files)


def resolve_model_path(model_name: str | None) -> str | None:
    if not model_name:
        return None
    return os.path.join(get_models_dir(), model_name)
