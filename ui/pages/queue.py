from __future__ import annotations

import reflex as rx

from ui.pages.common import page_shell
from ui.state.queue_state import QueueState


def _queue_row(row: dict[str, str]) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.code(row["id"]),
            rx.text(row["name"]),
            rx.text(row["status"]),
            rx.text(row["progress"]),
            spacing="3",
            width="100%",
        ),
        rx.text(f"Output: {row['output_name']}"),
        rx.text(row["desc"]),
        rx.divider(),
        width="100%",
        align="start",
    )


def queue_page() -> rx.Component:
    return page_shell(
        "Queue",
        rx.text("バックグラウンドで処理されるマージタスクを確認します。"),
        rx.hstack(
            rx.button("Refresh", on_click=QueueState.refresh),
            rx.button("Pause", on_click=QueueState.pause),
            rx.button("Resume", on_click=QueueState.resume),
            rx.button("Clear Completed", on_click=QueueState.clear_completed_items),
            spacing="3",
        ),
        rx.text(f"Paused: {QueueState.queue_paused}"),
        rx.text(QueueState.status_message),
        rx.hstack(
            rx.input(
                placeholder="Task ID to remove",
                value=QueueState.selected_task_id,
                on_change=QueueState.set_selected_task_id,
                width="100%",
            ),
            rx.button("Remove Task", on_click=QueueState.remove_selected_task),
            width="100%",
            spacing="3",
        ),
        rx.vstack(
            rx.foreach(QueueState.queue_rows, _queue_row),
            width="100%",
            align="start",
        ),
        on_mount=QueueState.load_page,
    )
