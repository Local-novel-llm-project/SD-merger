from __future__ import annotations

import reflex as rx

from ui.theme import THEME_TOKENS


FEEDBACK_VARIANTS = ("info", "success", "warning", "error")

FEEDBACK_STYLES = {
    "info": {
        "background": THEME_TOKENS["info_background"],
        "border_color": THEME_TOKENS["info_border"],
        "text_color": THEME_TOKENS["info_text"],
        "label": "Info",
    },
    "success": {
        "background": THEME_TOKENS["success_background"],
        "border_color": THEME_TOKENS["success_border"],
        "text_color": THEME_TOKENS["success_text"],
        "label": "Success",
    },
    "warning": {
        "background": THEME_TOKENS["warning_background"],
        "border_color": THEME_TOKENS["warning_border"],
        "text_color": THEME_TOKENS["warning_text"],
        "label": "Warning",
    },
    "error": {
        "background": THEME_TOKENS["error_background"],
        "border_color": THEME_TOKENS["error_border"],
        "text_color": THEME_TOKENS["error_text"],
        "label": "Error",
    },
}


def feedback_style(variant: str) -> dict[str, str]:
    return FEEDBACK_STYLES.get(variant, FEEDBACK_STYLES["info"])


def _feedback_box(message: str | rx.Var[str], variant: str) -> rx.Component:
    style = feedback_style(variant)
    return rx.box(
        rx.vstack(
            rx.badge(style["label"], color_scheme="gray", variant="soft"),
            rx.text(message, color=style["text_color"], size="3", width="100%"),
            spacing="2",
            align="start",
            width="100%",
        ),
        width="100%",
        padding="1rem 1.1rem",
        background=style["background"],
        border=f"1px solid {style['border_color']}",
        border_radius="14px",
        box_shadow=THEME_TOKENS["panel_shadow"],
    )


def feedback_banner(
    message: str | rx.Var[str],
    variant: str | rx.Var[str] = "info",
) -> rx.Component:
    return rx.cond(
        message != "",
        rx.box(
            rx.cond(
                variant == "success",
                _feedback_box(message, "success"),
                rx.cond(
                    variant == "warning",
                    _feedback_box(message, "warning"),
                    rx.cond(
                        variant == "error",
                        _feedback_box(message, "error"),
                        _feedback_box(message, "info"),
                    ),
                ),
            ),
            width="100%",
        ),
        rx.fragment(),
    )
