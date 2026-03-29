from __future__ import annotations

import reflex as rx

from ui.services.app_boot import ensure_app_ready
from ui.services.queue_service import (
    build_queue_rows,
    clear_completed,
    is_paused,
    pause_queue,
    remove_task,
    resume_queue,
)


class QueueState(rx.State):
    queue_rows: list[dict[str, str]] = []
    queue_paused: bool = False
    selected_task_id: str = ""
    status_message: str = ""
    status_variant: str = "info"

    def load_page(self) -> None:
        ensure_app_ready()
        self.refresh()

    def set_selected_task_id(self, value: str) -> None:
        self.selected_task_id = value

    def refresh(self) -> None:
        self.queue_rows = build_queue_rows()
        self.queue_paused = is_paused()

    def pause(self) -> None:
        pause_queue()
        self.status_message = "Queue paused."
        self.status_variant = "info"
        self.refresh()

    def resume(self) -> None:
        resume_queue()
        self.status_message = "Queue resumed."
        self.status_variant = "success"
        self.refresh()

    def clear_completed_items(self) -> None:
        clear_completed()
        self.status_message = "Completed and errored tasks were cleared."
        self.status_variant = "success"
        self.refresh()

    def remove_selected_task(self) -> None:
        if not self.selected_task_id.strip():
            self.status_message = "Task ID を入力してください。"
            self.status_variant = "error"
            return

        if remove_task(self.selected_task_id.strip()):
            self.status_message = f"Removed task: {self.selected_task_id.strip()}"
            self.status_variant = "success"
        else:
            self.status_message = (
                f"Task could not be removed: {self.selected_task_id.strip()}"
            )
            self.status_variant = "error"
        self.refresh()
