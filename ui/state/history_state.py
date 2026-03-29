from __future__ import annotations

import reflex as rx

from ui.services.app_boot import ensure_app_ready
from ui.services.history_service import (
    build_history_rows,
    build_history_yaml,
    rerun_history_entry,
)


class HistoryState(rx.State):
    history_rows: list[dict[str, str]] = []
    selected_output_name: str = ""
    yaml_preview: str = "# Select an output name to inspect its recipe."
    status_message: str = ""

    def load_page(self) -> None:
        ensure_app_ready()
        self.refresh()

    def refresh(self) -> None:
        self.history_rows = build_history_rows()

    def set_selected_output_name(self, value: str) -> None:
        self.selected_output_name = value

    def load_yaml_preview(self) -> None:
        if not self.selected_output_name.strip():
            self.status_message = "Output Name を入力してください。"
            return
        self.yaml_preview = build_history_yaml(self.selected_output_name.strip())
        self.status_message = f"Loaded recipe: {self.selected_output_name.strip()}"

    def rerun_selected(self) -> None:
        if not self.selected_output_name.strip():
            self.status_message = "Output Name を入力してください。"
            return
        task_id = rerun_history_entry(self.selected_output_name.strip())
        self.status_message = f"Queued rerun task: {task_id}"
        self.refresh()
