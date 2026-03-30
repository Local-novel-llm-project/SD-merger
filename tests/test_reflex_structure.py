from __future__ import annotations

import importlib
import sys
import types

from ui.navigation import get_navigation_items


def test_navigation_items_cover_expected_routes():
    items = get_navigation_items()

    assert [item.route for item in items] == ["/", "/queue", "/history", "/tune"]
    assert [item.label for item in items] == [
        "Merge",
        "Queue",
        "History",
        "Arthemy Tuner",
    ]


def test_app_theme_has_expected_tokens(monkeypatch):
    fake_reflex = types.ModuleType("reflex")
    fake_reflex.Component = object
    fake_reflex.theme = lambda **kwargs: {"theme": kwargs}

    monkeypatch.setitem(sys.modules, "reflex", fake_reflex)
    sys.modules.pop("ui.theme", None)
    theme_module = importlib.import_module("ui.theme")

    assert theme_module.APP_THEME["appearance"] == "light"
    assert theme_module.APP_THEME["accent_color"] == "amber"
    assert theme_module.APP_THEME["radius"] == "large"
    assert theme_module.APP_THEME["scaling"] == "105%"


def test_feedback_style_falls_back_to_info(monkeypatch):
    fake_reflex = types.ModuleType("reflex")
    fake_reflex.Component = object
    fake_reflex.Var = str

    monkeypatch.setitem(sys.modules, "reflex", fake_reflex)
    sys.modules.pop("ui.components.feedback", None)
    feedback_module = importlib.import_module("ui.components.feedback")

    assert feedback_module.feedback_style("success")["label"] == "Success"
    assert feedback_module.feedback_style("unknown") == feedback_module.feedback_style("info")


def test_build_app_registers_navigation_routes(monkeypatch):
    fake_reflex = types.ModuleType("reflex")

    class FakeApp:
        def __init__(self, theme=None):
            self.theme = theme
            self.pages = []

        def add_page(self, component, route, title):
            self.pages.append((component, route, title))

    fake_reflex.App = FakeApp
    fake_reflex.Component = object
    fake_reflex.theme = lambda **kwargs: {"theme": kwargs}

    page_modules = {
        "ui.pages.merge": ("merge_page", object()),
        "ui.pages.queue": ("queue_page", object()),
        "ui.pages.history": ("history_page", object()),
        "ui.pages.tune": ("tune_page", object()),
    }

    monkeypatch.setitem(sys.modules, "reflex", fake_reflex)
    for module_name, (attribute_name, value) in page_modules.items():
        module = types.ModuleType(module_name)
        setattr(module, attribute_name, value)
        monkeypatch.setitem(sys.modules, module_name, module)

    sys.modules.pop("ui.theme", None)
    sys.modules.pop("app.app", None)
    sys.modules.pop("ui.ui", None)
    app_module = importlib.import_module("app.app")
    ui_module = importlib.import_module("ui.ui")

    assert [route for _, route, _ in app_module.app.pages] == [
        "/",
        "/queue",
        "/history",
        "/tune",
    ]
    assert [title for _, _, title in app_module.app.pages] == [
        "SD-merger",
        "Queue",
        "History",
        "Arthemy Tuner",
    ]
    assert ui_module.app is app_module.app


def test_rxconfig_uses_top_level_app_entrypoint():
    import rxconfig

    assert rxconfig.config.app_name == "app"
    assert rxconfig.config.frontend_port == 3000
    assert rxconfig.config.backend_port == 3001
