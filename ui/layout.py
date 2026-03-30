from __future__ import annotations

import reflex as rx

from ui.components.feedback import feedback_banner
from ui.navigation import NavigationItem, get_navigation_items
from ui.theme import THEME_TOKENS


STATUS_BADGE_STYLES = {
    "pending": {
        "background": THEME_TOKENS["status_pending_background"],
        "border_color": THEME_TOKENS["status_pending_border"],
        "text_color": THEME_TOKENS["status_pending_text"],
        "label": "Pending",
    },
    "running": {
        "background": THEME_TOKENS["status_running_background"],
        "border_color": THEME_TOKENS["status_running_border"],
        "text_color": THEME_TOKENS["status_running_text"],
        "label": "Running",
    },
    "completed": {
        "background": THEME_TOKENS["status_completed_background"],
        "border_color": THEME_TOKENS["status_completed_border"],
        "text_color": THEME_TOKENS["status_completed_text"],
        "label": "Completed",
    },
    "success": {
        "background": THEME_TOKENS["status_completed_background"],
        "border_color": THEME_TOKENS["status_completed_border"],
        "text_color": THEME_TOKENS["status_completed_text"],
        "label": "Success",
    },
    "error": {
        "background": THEME_TOKENS["status_error_background"],
        "border_color": THEME_TOKENS["status_error_border"],
        "text_color": THEME_TOKENS["status_error_text"],
        "label": "Error",
    },
    "failed": {
        "background": THEME_TOKENS["status_error_background"],
        "border_color": THEME_TOKENS["status_error_border"],
        "text_color": THEME_TOKENS["status_error_text"],
        "label": "Failed",
    },
    "info": {
        "background": THEME_TOKENS["info_background"],
        "border_color": THEME_TOKENS["info_border"],
        "text_color": THEME_TOKENS["info_text"],
        "label": "Info",
    },
}


def _surface_box(*children: rx.Component, **props) -> rx.Component:
    return rx.box(
        *children,
        background=THEME_TOKENS["panel_background"],
        border=f"1px solid {THEME_TOKENS['panel_border']}",
        border_radius=THEME_TOKENS["radius"],
        box_shadow=THEME_TOKENS["panel_shadow"],
        backdrop_filter="blur(12px)",
        **props,
    )


def status_badge(status: str | rx.Var[str]) -> rx.Component:
    def _badge(label: str | rx.Var[str], style_key: str) -> rx.Component:
        style = STATUS_BADGE_STYLES[style_key]
        return rx.badge(
            label,
            variant="soft",
            color_scheme="gray",
            padding_x="0.5rem",
            padding_y="0.2rem",
            border=f"1px solid {style['border_color']}",
            background=style["background"],
            color=style["text_color"],
        )

    normalized = str(status).strip().lower()
    if normalized in STATUS_BADGE_STYLES:
        return _badge(STATUS_BADGE_STYLES[normalized]["label"], normalized)
    return rx.cond(
        status == "pending",
        _badge("Pending", "pending"),
        rx.cond(
            status == "running",
            _badge("Running", "running"),
            rx.cond(
                status == "completed",
                _badge("Completed", "completed"),
                rx.cond(
                    status == "Success",
                    _badge("Success", "success"),
                    rx.cond(
                        status == "error",
                        _badge("Error", "error"),
                        _badge(status, "info"),
                    ),
                ),
            ),
        ),
    )


def meta_badge(label: str | rx.Var[str]) -> rx.Component:
    return rx.badge(
        label,
        variant="soft",
        color_scheme="gray",
        background=THEME_TOKENS["accent_soft"],
        color=THEME_TOKENS["text_strong"],
        border=f"1px solid {THEME_TOKENS['accent_border']}",
        padding_x="0.65rem",
        padding_y="0.25rem",
    )


def section_card(
    *children: rx.Component,
    title: str | None = None,
    description: str | None = None,
    actions: rx.Component | None = None,
) -> rx.Component:
    header = (
        rx.hstack(
            rx.vstack(
                rx.heading(title, size="4", color=THEME_TOKENS["text_strong"]),
                rx.cond(
                    bool(description),
                    rx.text(
                        description or "",
                        color=THEME_TOKENS["text_muted"],
                        size="2",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                spacing="1",
                align="start",
                width="100%",
            ),
            actions if actions is not None else rx.fragment(),
            justify="between",
            align="start",
            width="100%",
            gap="3",
            wrap="wrap",
        )
        if title
        else rx.fragment()
    )
    return _surface_box(
        rx.vstack(
            header,
            *children,
            spacing="4",
            align="start",
            width="100%",
        ),
        width="100%",
        padding={"initial": "1rem", "md": "1.25rem"},
    )


def page_header(
    title: str,
    description: str | None = None,
    meta_items: list[str | rx.Var[str]] | None = None,
    actions: rx.Component | None = None,
) -> rx.Component:
    badges = meta_items or []
    return _surface_box(
        rx.vstack(
            rx.badge(
                "Reflex Workflow",
                variant="soft",
                color_scheme="gray",
                background=THEME_TOKENS["accent_soft"],
                color=THEME_TOKENS["text_strong"],
                border=f"1px solid {THEME_TOKENS['accent_border']}",
            ),
            rx.hstack(
                rx.vstack(
                    rx.heading(title, size="7", color=THEME_TOKENS["text_strong"]),
                    rx.cond(
                        bool(description),
                        rx.text(
                            description or "",
                            color=THEME_TOKENS["text_muted"],
                            size="3",
                            width="100%",
                        ),
                        rx.fragment(),
                    ),
                    spacing="2",
                    align="start",
                    width="100%",
                ),
                actions if actions is not None else rx.fragment(),
                justify="between",
                align="start",
                width="100%",
                gap="4",
                wrap="wrap",
            ),
            rx.cond(
                bool(badges),
                rx.flex(
                    *[meta_badge(item) for item in badges],
                    wrap="wrap",
                    gap="2",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        width="100%",
        padding={"initial": "1.1rem", "md": "1.35rem"},
        background=THEME_TOKENS["panel_alt_background"],
    )


def status_strip(
    message: str | rx.Var[str],
    variant: str | rx.Var[str] = "info",
    busy_message: str | rx.Var[str] = "",
) -> rx.Component:
    busy_box = rx.cond(
        busy_message != "",
        _surface_box(
            rx.hstack(
                rx.spinner(size="2"),
                rx.vstack(
                    rx.text("Working", weight="bold", color=THEME_TOKENS["text_strong"]),
                    rx.text(busy_message, color=THEME_TOKENS["text_muted"], width="100%"),
                    spacing="1",
                    align="start",
                    width="100%",
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            width="100%",
            padding="0.95rem 1.1rem",
            background=THEME_TOKENS["panel_alt_background"],
        ),
        rx.fragment(),
    )
    feedback_box = feedback_banner(message, variant)
    return rx.vstack(
        busy_box,
        feedback_box,
        spacing="3",
        width="100%",
        align="start",
    )


def log_panel(
    value: str | rx.Var[str],
    *,
    min_height: str = "18rem",
    read_only: bool = True,
    on_change=None,
) -> rx.Component:
    return rx.text_area(
        value=value,
        read_only=read_only,
        on_change=on_change,
        min_height=min_height,
        width="100%",
        background="rgba(255, 251, 245, 0.95)",
        color=THEME_TOKENS["text_strong"],
        border=f"1px solid {THEME_TOKENS['panel_border']}",
        border_radius=THEME_TOKENS["radius_small"],
        font_family="'IBM Plex Mono', 'Noto Sans Mono', monospace",
        font_size="0.9rem",
        line_height="1.5",
        padding="0.9rem",
    )


def record_row(
    title: str | rx.Var[str],
    *,
    status: str | rx.Var[str] = "",
    subtitle: str | rx.Var[str] = "",
    meta: list[rx.Component] | None = None,
    action: rx.Component | None = None,
) -> rx.Component:
    meta_items = meta or []
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text(title, weight="medium", color=THEME_TOKENS["text_strong"]),
                    rx.cond(
                        subtitle != "",
                        rx.text(
                            subtitle,
                            color=THEME_TOKENS["text_muted"],
                            size="2",
                            width="100%",
                        ),
                        rx.fragment(),
                    ),
                    spacing="1",
                    align="start",
                    width="100%",
                ),
                action if action is not None else rx.fragment(),
                justify="between",
                align="start",
                width="100%",
                gap="3",
                wrap="wrap",
            ),
            rx.flex(
                rx.cond(status != "", status_badge(status), rx.fragment()),
                *meta_items,
                wrap="wrap",
                gap="2",
                width="100%",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        width="100%",
        padding="1rem",
        background="rgba(255, 251, 245, 0.92)",
        border=f"1px solid {THEME_TOKENS['panel_border']}",
        border_radius=THEME_TOKENS["radius_small"],
    )


def _nav_link(item: NavigationItem, current_route: str) -> rx.Component:
    is_active = current_route == item.route
    active_background = THEME_TOKENS["accent_soft"]
    idle_background = "rgba(255, 255, 255, 0.72)"
    return rx.link(
        rx.vstack(
            rx.hstack(
                rx.text(
                    item.label,
                    weight="bold",
                    color=THEME_TOKENS["text_strong"],
                ),
                rx.cond(is_active, meta_badge("Current"), rx.fragment()),
                justify="between",
                width="100%",
                align="center",
            ),
            rx.text(
                item.description,
                size="2",
                color=THEME_TOKENS["text_muted"],
                width="100%",
            ),
            spacing="2",
            align="start",
            width="100%",
        ),
        href=item.route,
        underline="none",
        width="100%",
        padding="0.95rem 1rem",
        background=active_background if is_active else idle_background,
        border=(
            f"1px solid {THEME_TOKENS['accent_border']}"
            if is_active
            else f"1px solid {THEME_TOKENS['panel_border']}"
        ),
        border_radius=THEME_TOKENS["radius_small"],
        box_shadow="0 10px 24px rgba(94, 62, 30, 0.06)",
    )


def navigation_menu(current_route: str) -> rx.Component:
    return rx.vstack(
        *[_nav_link(item, current_route) for item in get_navigation_items()],
        spacing="3",
        width="100%",
        align="start",
    )


def sidebar_panel(current_route: str) -> rx.Component:
    return _surface_box(
        rx.vstack(
            rx.vstack(
                rx.text("SD-merger", size="2", color=THEME_TOKENS["text_soft"]),
                rx.heading("Workflow Console", size="6", color=THEME_TOKENS["text_strong"]),
                rx.text(
                    "Merge、Queue、History、Tuner を同じ導線で扱える共通 UI です。",
                    color=THEME_TOKENS["text_muted"],
                    width="100%",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            navigation_menu(current_route),
            spacing="5",
            align="start",
            width="100%",
            position={"initial": "relative", "lg": "sticky"},
            top={"lg": "1.5rem"},
        ),
        width="100%",
        padding={"initial": "1rem", "md": "1.2rem"},
        background=THEME_TOKENS["sidebar_background"],
    )


def page_shell(
    title: str,
    *children: rx.Component,
    current_route: str,
    description: str | None = None,
    feedback_message: str | rx.Var[str] = "",
    feedback_variant: str | rx.Var[str] = "info",
    busy_message: str | rx.Var[str] = "",
    meta_items: list[str | rx.Var[str]] | None = None,
    header_actions: rx.Component | None = None,
    **kwargs,
) -> rx.Component:
    return rx.box(
        rx.container(
            rx.flex(
                rx.box(
                    sidebar_panel(current_route),
                    width={"initial": "100%", "lg": "300px"},
                    flex_shrink="0",
                ),
                rx.vstack(
                    page_header(
                        title,
                        description=description,
                        meta_items=meta_items,
                        actions=header_actions,
                    ),
                    status_strip(
                        feedback_message,
                        feedback_variant,
                        busy_message=busy_message,
                    ),
                    *children,
                    spacing="5",
                    width="100%",
                    align="start",
                ),
                direction={"initial": "column", "lg": "row"},
                spacing="5",
                width="100%",
                align="start",
            ),
            max_width=THEME_TOKENS["content_width"],
            padding_x={"initial": "1rem", "md": "1.5rem"},
            padding_y={"initial": "1rem", "md": "1.5rem"},
            **kwargs,
        ),
        min_height="100vh",
        width="100%",
        background=THEME_TOKENS["page_background"],
        color=THEME_TOKENS["text_strong"],
        font_family=THEME_TOKENS["font_family"],
    )
