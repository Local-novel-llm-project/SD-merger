from __future__ import annotations

import reflex as rx

from ui.services.app_boot import ensure_app_ready
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


HISTORY_YAML_UPLOAD_ID = "history_yaml_upload"


class HistoryState(rx.State):
    history_rows: list[dict[str, str]] = []
    selected_output_name: str = ""
    yaml_preview: str = "# Select an output name to inspect its recipe."
    yaml_editor_text: str = ""
    status_message: str = ""
    status_variant: str = "info"
    import_path: str = ""
    export_path: str = ""
    yaml_source_label: str = ""

    def load_page(self) -> None:
        ensure_app_ready()
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

    def select_history_entry(self, output_name: str) -> None:
        self.selected_output_name = output_name
        self.load_yaml_preview()

    def load_yaml_preview(self) -> None:
        if not self.selected_output_name.strip():
            self.status_message = "Output Name を入力してください。"
            self.status_variant = "error"
            return

        output_name = self.selected_output_name.strip()
        self.yaml_preview = build_history_yaml(output_name)
        self.yaml_editor_text = self.yaml_preview
        self.yaml_source_label = f"History: {output_name}"
        self.status_message = f"Loaded recipe: {output_name}"
        self.status_variant = "success"

    def validate_yaml_editor(self) -> None:
        parse_history_yaml_text(self.yaml_editor_text)
        self.status_message = "YAML validation succeeded."
        self.status_variant = "success"

    async def handle_yaml_upload(self, files: list[rx.UploadFile]):
        if not files:
            self.status_message = "YAML file was not selected."
            self.status_variant = "error"
            return None

        upload = files[0]
        contents = await upload.read()
        self.yaml_editor_text = contents.decode("utf-8")
        self.yaml_preview = self.yaml_editor_text
        self.yaml_source_label = f"Upload: {upload.filename}"
        self.status_message = f"Imported YAML from upload: {upload.filename}"
        self.status_variant = "success"
        return rx.clear_selected_files(HISTORY_YAML_UPLOAD_ID)

    def import_yaml_from_path(self) -> None:
        yaml_text = load_yaml_from_path(self.import_path)
        self.yaml_editor_text = yaml_text
        self.yaml_preview = yaml_text
        self.yaml_source_label = f"Path: {self.import_path.strip()}"
        self.status_message = f"Imported YAML from path: {self.import_path.strip()}"
        self.status_variant = "success"

    def export_selected_history_to_path(self) -> None:
        if not self.selected_output_name.strip():
            self.status_message = "Output Name を入力してください。"
            self.status_variant = "error"
            return

        exported_path = export_history_entry_to_path(
            self.selected_output_name.strip(),
            self.export_path,
        )
        self.status_message = f"Exported history YAML: {exported_path}"
        self.status_variant = "success"

    def export_editor_to_path(self) -> None:
        exported_path = export_yaml_to_path(self.yaml_editor_text, self.export_path)
        self.status_message = f"Exported edited YAML: {exported_path}"
        self.status_variant = "success"

    def download_yaml(self):
        data, filename = build_yaml_download_payload(
            self.yaml_editor_text,
            self.selected_output_name or self.yaml_source_label,
        )
        self.status_message = f"Downloading YAML: {filename}"
        self.status_variant = "info"
        return rx.download(data=data, filename=filename)

    def rerun_selected(self) -> None:
        fallback_output_name = self.selected_output_name.strip() or "imported_recipe"
        task_id, resolved_output_name = queue_history_yaml(
            self.yaml_editor_text,
            fallback_output_name=fallback_output_name,
        )
        self.selected_output_name = resolved_output_name
        self.status_message = f"Queued rerun task: {task_id}"
        self.status_variant = "success"
        self.refresh()
