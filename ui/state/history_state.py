from __future__ import annotations

import reflex as rx

from ui.services.history_service import (
    build_yaml_download_payload,
    build_history_rows,
    build_history_yaml,
    export_history_entry_to_path,
    export_yaml_to_path,
    find_history_entry,
    load_yaml_from_path,
    parse_history_yaml_text,
    queue_history_yaml,
)
from ui.state.base import BasePageState


HISTORY_YAML_UPLOAD_ID = "history_yaml_upload"


class HistoryState(BasePageState):
    history_rows: list[dict[str, str]] = []
    selected_output_name: str = ""
    yaml_preview: str = "# Select an output name to inspect its recipe."
    yaml_editor_text: str = ""
    import_path: str = ""
    export_path: str = ""
    yaml_source_label: str = ""

    def load_page(self) -> None:
        self.ensure_ready()
        self.refresh()

    def refresh(self) -> None:
        self.history_rows = build_history_rows()
        if self.selected_output_name and not find_history_entry(self.selected_output_name):
            self.selected_output_name = ""

    def set_selected_output_name(self, value: str) -> None:
        self.selected_output_name = value

    def set_import_path(self, value: str) -> None:
        self.import_path = value

    def set_export_path(self, value: str) -> None:
        self.export_path = value

    def set_yaml_editor_text(self, value: str) -> None:
        self.yaml_editor_text = value

    @rx.event
    def select_history_entry(self, output_name: str) -> None:
        self.selected_output_name = output_name
        self.load_yaml_preview()

    def load_yaml_preview(self) -> None:
        if not self.selected_output_name.strip():
            self.set_status("Output Name を入力してください。", "error")
            return

        self.begin_busy("履歴レシピを読み込んでいます。")
        try:
            output_name = self.selected_output_name.strip()
            self.yaml_preview = build_history_yaml(output_name)
            self.yaml_editor_text = self.yaml_preview
            self.yaml_source_label = f"History: {output_name}"
            self.end_busy(f"Loaded recipe: {output_name}")
        except Exception as exc:
            self.fail_busy(exc, action="履歴レシピの読込")

    def validate_yaml_editor(self) -> None:
        self.begin_busy("YAML を検証しています。")
        try:
            parse_history_yaml_text(self.yaml_editor_text)
            self.end_busy("YAML validation succeeded.")
        except Exception as exc:
            self.fail_busy(exc, action="YAML 検証")

    @rx.event
    async def handle_yaml_upload(self, files: list[rx.UploadFile]):
        self.begin_busy("YAML をアップロードしています。")
        try:
            yaml_text, filename, clear_action = await self.read_uploaded_text(
                files,
                upload_id=HISTORY_YAML_UPLOAD_ID,
                empty_message="YAML file was not selected.",
            )
            if yaml_text is None or filename is None:
                self.busy = False
                self.busy_message = ""
                return None

            self.yaml_editor_text = yaml_text
            self.yaml_preview = yaml_text
            self.yaml_source_label = f"Upload: {filename}"
            self.end_busy(f"Imported YAML from upload: {filename}")
            return clear_action
        except Exception as exc:
            self.fail_busy(exc, action="YAML アップロード")
            return None

    def import_yaml_from_path(self) -> None:
        self.begin_busy("YAML をサーバーパスから読み込んでいます。")
        try:
            yaml_text = load_yaml_from_path(self.import_path)
            self.yaml_editor_text = yaml_text
            self.yaml_preview = yaml_text
            self.yaml_source_label = f"Path: {self.import_path.strip()}"
            self.end_busy(f"Imported YAML from path: {self.import_path.strip()}")
        except Exception as exc:
            self.fail_busy(exc, action="YAML のパス読込")

    def export_selected_history_to_path(self) -> None:
        if not self.selected_output_name.strip():
            self.set_status("Output Name を入力してください。", "error")
            return

        self.begin_busy("履歴 YAML を書き出しています。")
        try:
            exported_path = export_history_entry_to_path(
                self.selected_output_name.strip(),
                self.export_path,
            )
            self.end_busy(f"Exported history YAML: {exported_path}")
        except Exception as exc:
            self.fail_busy(exc, action="履歴 YAML の書き出し")

    def export_editor_to_path(self) -> None:
        self.begin_busy("編集中の YAML を書き出しています。")
        try:
            exported_path = export_yaml_to_path(self.yaml_editor_text, self.export_path)
            self.end_busy(f"Exported edited YAML: {exported_path}")
        except Exception as exc:
            self.fail_busy(exc, action="編集中 YAML の書き出し")

    def download_yaml(self):
        try:
            data, filename = build_yaml_download_payload(
                self.yaml_editor_text,
                self.selected_output_name or self.yaml_source_label,
            )
            return self.build_download(data, filename, message=f"Downloading YAML: {filename}")
        except Exception as exc:
            self.set_error(exc, action="YAML ダウンロード")
            return None

    def rerun_selected(self) -> None:
        self.begin_busy("履歴レシピを再投入しています。")
        try:
            fallback_output_name = self.selected_output_name.strip() or "imported_recipe"
            task_id, resolved_output_name = queue_history_yaml(
                self.yaml_editor_text,
                fallback_output_name=fallback_output_name,
            )
            self.selected_output_name = resolved_output_name
            self.refresh()
            self.end_busy(f"Queued rerun task: {task_id}")
        except Exception as exc:
            self.fail_busy(exc, action="履歴レシピの再投入")
