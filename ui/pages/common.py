from __future__ import annotations

import reflex as rx

from ui.components.feedback import feedback_banner
from ui.layout import navigation_menu, page_shell, section_card
from ui.navigation import get_navigation_items


def nav_bar(current_route: str = "/") -> rx.Component:
    return navigation_menu(current_route)


__all__ = [
    "feedback_banner",
    "get_navigation_items",
    "nav_bar",
    "page_shell",
    "section_card",
]
