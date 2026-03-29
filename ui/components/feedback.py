from __future__ import annotations

import reflex as rx


FEEDBACK_VARIANTS = ("info", "success", "warning", "error")

FEEDBACK_STYLES = {
    "info": {
        "background": "#fff7e6",
        "border_color": "#f59e0b",
        "text_color": "#92400e",
        "label": "Info",
    },
    "success": {
        "background": "#eefbf3",
        "border_color": "#16a34a",
        "text_color": "#166534",
        "label": "Success",
    },
    "warning": {
        "background": "#fff4e5",
        "border_color": "#ea580c",
        "text_color": "#9a3412",
        "label": "Warning",
    },
    "error": {
        "background": "#fff1f2",
        "border_color": "#e11d48",
        "text_color": "#9f1239",
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
        box_shadow="0 10px 24px rgba(15, 23, 42, 0.05)",
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
