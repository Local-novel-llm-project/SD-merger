from __future__ import annotations

import reflex as rx

from ui.pages.common import meta_badge, page_shell, record_row, section_card
from ui.state.queue_state import QueueState


def _queue_row(row: dict[str, str]) -> rx.Component:
    return record_row(
        row["name"],
        status=row["status"],
        subtitle=row["desc"],
        meta=[
            meta_badge(row["id"]),
            meta_badge(row["progress"]),
            meta_badge(f"Output: {row['output_name']}"),
        ],
        action=rx.button(
            "Select",
            size="1",
            variant="soft",
            on_click=QueueState.select_task(row["id"]),
        ),
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
            title="Queue Controls",
            description="実行状態の確認、一時停止、再開、完了タスク整理、個別削除を扱います。",
        ),
        section_card(
            rx.vstack(
                rx.foreach(QueueState.queue_rows, _queue_row),
                width="100%",
                align="start",
                spacing="3",
            ),
            title="Tasks",
            description="タスクの状態、進行状況、出力先を一覧で確認できます。",
        ),
        current_route="/queue",
        description="実行中タスクの一覧、停止状態、削除操作をひとつの導線で扱います。",
        feedback_message=QueueState.status_message,
        feedback_variant=QueueState.status_variant,
        busy_message=QueueState.busy_message,
        header_actions=rx.vstack(
            rx.text(f"Task Count: {QueueState.task_count}", color="#6a5b4d", size="2"),
            rx.text(f"Paused: {QueueState.queue_paused}", color="#6a5b4d", size="2"),
            rx.text(f"Last Updated: {QueueState.last_updated_at}", color="#8b7a68", size="2"),
            spacing="1",
            align="end",
        ),
        on_mount=QueueState.load_page,
    )
