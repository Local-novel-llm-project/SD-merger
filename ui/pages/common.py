from __future__ import annotations

import reflex as rx

from ui.components.feedback import feedback_banner
from ui.layout import (
    log_panel,
    meta_badge,
    navigation_menu,
    page_header,
    page_shell,
    record_row,
    section_card,
    status_badge,
    status_strip,
)
from ui.navigation import get_navigation_items


def nav_bar(current_route: str = "/") -> rx.Component:
    return navigation_menu(current_route)


__all__ = [
    "feedback_banner",
    "get_navigation_items",
    "log_panel",
    "meta_badge",
    "nav_bar",
    "page_header",
    "page_shell",
    "record_row",
    "section_card",
    "status_badge",
    "status_strip",
]
