from __future__ import annotations

import reflex as rx

from ui.navigation import get_navigation_items
from ui.pages.history import history_page
from ui.pages.merge import merge_page
from ui.pages.queue import queue_page
from ui.pages.tune import tune_page
from ui.theme import create_theme


PAGE_COMPONENTS = {
    "/": merge_page,
    "/queue": queue_page,
    "/history": history_page,
    "/tune": tune_page,
}


def build_app() -> rx.App:
    app = rx.App(theme=create_theme())
    for item in get_navigation_items():
        app.add_page(PAGE_COMPONENTS[item.route], route=item.route, title=item.title)
    return app


app = build_app()
