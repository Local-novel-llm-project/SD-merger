from __future__ import annotations

import asyncio
from datetime import datetime

import reflex as rx

from ui.services.queue_service import (
    build_queue_snapshot,
    clear_completed,
    get_queue_task,
    pause_queue,
    remove_task,
    resume_queue,
)
from ui.state.base import BasePageState


class QueueState(BasePageState):
    queue_rows: list[dict[str, str]] = []
    queue_paused: bool = False
    selected_task_id: str = ""
    task_count: int = 0
    last_updated_at: str = ""
    poll_route: str = "/queue"

    def load_page(self) -> None:
        self.ensure_ready()
        self.refresh()
        return self.start_polling()

    def set_selected_task_id(self, value: str) -> None:
        self.selected_task_id = value

    @rx.event
    def select_task(self, task_id: str) -> None:
        self.selected_task_id = task_id

    @rx.var
    def selected_task_summary(self) -> str:
        if not self.selected_task_id:
            return "No task selected."
        task = get_queue_task(self.selected_task_id)
        if task is None:
            return f"Task not found: {self.selected_task_id}"
        return (
            f"{task.get('name', '')} / {task.get('status', '')} / "
            f"{task.get('progress_desc', '')}"
        )

    def _refresh_snapshot(self) -> None:
        snapshot = build_queue_snapshot()
        self.queue_rows = snapshot["rows"]
        self.queue_paused = snapshot["paused"]
        self.task_count = snapshot["task_count"]
        self.last_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def refresh(self) -> None:
        self._refresh_snapshot()

    def pause(self) -> None:
        self.begin_busy("キューを一時停止しています。")
        try:
            _, message = pause_queue()
            self.refresh()
            self.end_busy(message, variant="info")
        except Exception as exc:
            self.fail_busy(exc, action="キューの一時停止")

    def resume(self) -> None:
        self.begin_busy("キューを再開しています。")
        try:
            _, message = resume_queue()
            self.refresh()
            self.end_busy(message)
        except Exception as exc:
            self.fail_busy(exc, action="キューの再開")

    def clear_completed_items(self) -> None:
        self.begin_busy("完了済みタスクを整理しています。")
        try:
            _, message = clear_completed()
            self.refresh()
            self.end_busy(message)
        except Exception as exc:
            self.fail_busy(exc, action="完了済みタスクの削除")

    def remove_selected_task(self) -> None:
        if not self.selected_task_id.strip():
            self.set_status("Task ID を入力してください。", "error")
            return

        self.begin_busy("タスクを削除しています。")
        try:
            ok, message = remove_task(self.selected_task_id.strip())
            self.refresh()
            self.end_busy(message, variant="success" if ok else "error")
        except Exception as exc:
            self.fail_busy(exc, action="タスクの削除")

    def _poll_tick(self) -> None:
        self._refresh_snapshot()
