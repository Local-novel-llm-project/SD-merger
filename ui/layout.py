from __future__ import annotations

import reflex as rx

from ui.components.feedback import feedback_banner
from ui.navigation import NavigationItem, get_navigation_items


def section_card(*children: rx.Component, title: str | None = None) -> rx.Component:
    header = rx.heading(title, size="4") if title else rx.fragment()
    return rx.box(
        rx.vstack(
            header,
            *children,
            spacing="4",
            align="start",
            width="100%",
        ),
        width="100%",
        padding={"initial": "1rem", "md": "1.25rem"},
        background="rgba(255, 255, 255, 0.88)",
        border="1px solid rgba(148, 163, 184, 0.24)",
        border_radius="18px",
        box_shadow="0 18px 40px rgba(148, 163, 184, 0.12)",
        backdrop_filter="blur(10px)",
    )


def _nav_link(item: NavigationItem, current_route: str) -> rx.Component:
    is_active = current_route == item.route
    return rx.link(
        rx.vstack(
            rx.text(
                item.label,
                weight="medium",
                color="#1f2937" if is_active else "#475569",
            ),
            rx.text(item.description, size="1", color="#64748b", width="100%"),
            spacing="1",
            align="start",
            width="100%",
        ),
        href=item.route,
        underline="none",
        width="100%",
        padding="0.85rem 0.95rem",
        background="#fff7e6" if is_active else "rgba(255, 255, 255, 0.74)",
        border="1px solid #f59e0b" if is_active else "1px solid rgba(148, 163, 184, 0.24)",
        border_radius="16px",
        box_shadow="0 12px 28px rgba(148, 163, 184, 0.1)",
    )


def navigation_menu(current_route: str) -> rx.Component:
    return rx.grid(
        *[_nav_link(item, current_route) for item in get_navigation_items()],
        columns={"initial": "1", "md": "2", "lg": "4"},
        spacing="3",
        width="100%",
    )


def page_shell(
    title: str,
    *children: rx.Component,
    current_route: str,
    description: str | None = None,
    feedback_message: str | rx.Var[str] = "",
    feedback_variant: str | rx.Var[str] = "info",
    **kwargs,
) -> rx.Component:
    description_block = (
        rx.text(description, color="#475569", size="3", width="100%")
        if description
        else rx.fragment()
    )
    return rx.box(
        rx.container(
            rx.vstack(
                section_card(
                    rx.vstack(
                        rx.badge("Reflex App", variant="soft", color_scheme="gray"),
                        rx.heading(title, size="7", color="#1f2937"),
                        description_block,
                        spacing="3",
                        align="start",
                        width="100%",
                    ),
                    navigation_menu(current_route),
                ),
                feedback_banner(feedback_message, feedback_variant),
                *children,
                spacing="5",
                width="100%",
                align="start",
                padding_y="2rem",
            ),
            max_width="1280px",
            padding_x={"initial": "1rem", "md": "1.5rem"},
            **kwargs,
        ),
        min_height="100vh",
        width="100%",
        background=(
            "linear-gradient(180deg, rgba(255, 251, 235, 0.92) 0%, "
            "rgba(248, 250, 252, 1) 45%, rgba(255, 255, 255, 1) 100%)"
        ),
    )
