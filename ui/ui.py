from __future__ import annotations

import reflex as rx

from ui.pages.history import history_page
from ui.pages.merge import merge_page
from ui.pages.queue import queue_page
from ui.pages.tune import tune_page


def build_app() -> rx.App:
    app = rx.App()
    app.add_page(merge_page, route="/", title="SD-merger")
    app.add_page(queue_page, route="/queue", title="Queue")
    app.add_page(history_page, route="/history", title="History")
    app.add_page(tune_page, route="/tune", title="Arthemy Tuner")
    return app


app = build_app()
