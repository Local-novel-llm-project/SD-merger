from __future__ import annotations

import reflex as rx

from ui.pages.common import page_shell, section_card
from ui.state.queue_state import QueueState


def _queue_row(row: dict[str, str]) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.code(row["id"]),
            rx.text(row["name"]),
            rx.text(row["status"]),
            rx.text(row["progress"]),
            rx.button(
                "Select",
                size="1",
                variant="soft",
                on_click=QueueState.select_task(row["id"]),
            ),
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
        section_card(
            rx.text("バックグラウンドで処理されるマージタスクを確認します。Queue 画面は自動更新されます。"),
            rx.hstack(
                rx.button("Refresh", on_click=QueueState.refresh, disabled=QueueState.busy),
                rx.button(
                    "Pause",
                    on_click=QueueState.pause,
                    disabled=QueueState.busy,
                    loading=QueueState.busy,
                ),
                rx.button(
                    "Resume",
                    on_click=QueueState.resume,
                    disabled=QueueState.busy,
                    loading=QueueState.busy,
                ),
                rx.button(
                    "Clear Completed",
                    on_click=QueueState.clear_completed_items,
                    disabled=QueueState.busy,
                    loading=QueueState.busy,
                ),
                spacing="3",
                wrap="wrap",
            ),
            rx.text(f"Paused: {QueueState.queue_paused}"),
            rx.text(f"Task Count: {QueueState.task_count}"),
            rx.text(f"Last Updated: {QueueState.last_updated_at}"),
            rx.text(QueueState.selected_task_summary),
            rx.hstack(
                rx.input(
                    placeholder="Task ID to remove",
                    value=QueueState.selected_task_id,
                    on_change=QueueState.set_selected_task_id,
                    disabled=QueueState.busy,
                    width="100%",
                ),
                rx.button(
                    "Remove Task",
                    on_click=QueueState.remove_selected_task,
                    disabled=QueueState.busy,
                    loading=QueueState.busy,
                ),
                width="100%",
                spacing="3",
            ),
            rx.cond(QueueState.busy_message != "", rx.text(QueueState.busy_message)),
            title="Queue Controls",
        ),
        section_card(
            rx.vstack(
                rx.foreach(QueueState.queue_rows, _queue_row),
                width="100%",
                align="start",
            ),
            title="Tasks",
        ),
        current_route="/queue",
        description="実行中タスクの一覧、停止状態、削除操作をひとつの導線で扱います。",
        feedback_message=QueueState.status_message,
        feedback_variant=QueueState.status_variant,
        on_mount=QueueState.load_page,
    )
