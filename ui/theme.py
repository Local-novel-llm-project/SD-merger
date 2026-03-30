from __future__ import annotations

import reflex as rx
from typing import Any, cast


APP_THEME = {
    "appearance": "light",
    "accent_color": "amber",
    "gray_color": "sand",
    "radius": "large",
    "scaling": "105%",
    "has_background": True,
}


def create_theme() -> rx.Component:
    return rx.theme(**cast(dict[str, Any], APP_THEME))
