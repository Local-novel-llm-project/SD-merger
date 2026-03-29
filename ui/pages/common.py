from __future__ import annotations

import reflex as rx


def nav_bar() -> rx.Component:
    return rx.hstack(
        rx.link("Merge", href="/"),
        rx.link("Queue", href="/queue"),
        rx.link("History", href="/history"),
        rx.link("Arthemy", href="/tune"),
        spacing="5",
        width="100%",
    )


def page_shell(title: str, *children: rx.Component) -> rx.Component:
    return rx.container(
        rx.vstack(
            nav_bar(),
            rx.heading(title),
            *children,
            spacing="6",
            width="100%",
            padding_y="2rem",
            align="start",
        ),
        max_width="1100px",
        padding_x="1.5rem",
    )
