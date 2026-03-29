from __future__ import annotations

from ui.utils import get_model_list, get_model_path


def list_models() -> list[str]:
    return get_model_list()


def resolve_model_path(model_name: str | None) -> str | None:
    return get_model_path(model_name)
