from __future__ import annotations

import asyncio
from datetime import datetime

import reflex as rx

from ui.services.app_boot import ensure_app_ready
from ui.services.queue_service import (
    build_queue_snapshot,
    clear_completed,
    get_queue_task,
    pause_queue,
    remove_task,
    resume_queue,
)


class QueueState(rx.State):
    queue_rows: list[dict[str, str]] = []
    queue_paused: bool = False
    selected_task_id: str = ""
    status_message: str = ""
    task_count: int = 0
    last_updated_at: str = ""
    polling_enabled: bool = False
    polling_generation: int = 0

    def load_page(self) -> None:
        ensure_app_ready()
        self.polling_enabled = True
        self.polling_generation += 1
        self.refresh()
        return QueueState.poll_queue(self.polling_generation)

    def set_selected_task_id(self, value: str) -> None:
        self.selected_task_id = value

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
        _, message = pause_queue()
        self.status_message = message
        self.refresh()

    def resume(self) -> None:
        _, message = resume_queue()
        self.status_message = message
        self.refresh()

    def clear_completed_items(self) -> None:
        _, message = clear_completed()
        self.status_message = message
        self.refresh()

    def remove_selected_task(self) -> None:
        if not self.selected_task_id.strip():
            self.status_message = "Task ID を入力してください。"
            return

        _, message = remove_task(self.selected_task_id.strip())
        self.status_message = message
        self.refresh()

    @rx.event(background=True)
    async def poll_queue(self, generation: int) -> None:
        while True:
            async with self:
                current_path = getattr(self.router.page, "path", "")
                if (
                    not self.polling_enabled
                    or self.polling_generation != generation
                    or current_path != "/queue"
                ):
                    self.polling_enabled = False
                    break
                self._refresh_snapshot()
            await asyncio.sleep(2)
