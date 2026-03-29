from __future__ import annotations

import reflex as rx

from ui.pages.common import page_shell, section_card
from ui.state.history_state import HistoryState


def _history_row(row: dict[str, str]) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(row["date"]),
            rx.code(row["output_name"]),
            rx.text(row["status"]),
            spacing="3",
            width="100%",
        ),
        rx.text(f"{row['model_a']}  <-  {row['model_b']}"),
        rx.text(
            f"Strategy: {row['strategy']} / Velocity: {row['velocity']} / LRV: {row['left_right_velocity']}"
        ),
        rx.divider(),
        width="100%",
        align="start",
    )


def history_page() -> rx.Component:
    return page_shell(
        "History",
        section_card(
            rx.text("過去の実行履歴を確認し、保存済みレシピから再実行できます。"),
            rx.hstack(
                rx.button("Refresh", on_click=HistoryState.refresh),
                rx.button("Load YAML Preview", on_click=HistoryState.load_yaml_preview),
                rx.button("Queue Rerun", on_click=HistoryState.rerun_selected),
                spacing="3",
                wrap="wrap",
            ),
            rx.input(
                placeholder="Output Name",
                value=HistoryState.selected_output_name,
                on_change=HistoryState.set_selected_output_name,
                width="100%",
            ),
            title="History Controls",
        ),
        section_card(
            rx.vstack(
                rx.foreach(HistoryState.history_rows, _history_row),
                width="100%",
                align="start",
            ),
            title="History Entries",
        ),
        section_card(
            rx.text_area(
                value=HistoryState.yaml_preview,
                read_only=True,
                min_height="24rem",
                width="100%",
            ),
            title="Recipe Preview",
        ),
        current_route="/history",
        description="実行履歴を参照し、保存済み YAML の確認と再投入を行います。",
        feedback_message=HistoryState.status_message,
        feedback_variant=HistoryState.status_variant,
        on_mount=HistoryState.load_page,
    )
