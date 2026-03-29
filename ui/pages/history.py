from __future__ import annotations

import reflex as rx

from ui.pages.common import page_shell, section_card
from ui.state.history_state import HISTORY_YAML_UPLOAD_ID, HistoryState


def _history_row(row: dict[str, str]) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(row["date"]),
            rx.code(row["output_name"]),
            rx.text(row["status"]),
            rx.button(
                "Load",
                size="1",
                variant="soft",
                on_click=HistoryState.select_history_entry(row["output_name"]),
            ),
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
            rx.text("過去の実行履歴を確認し、YAML を編集・import/export して再実行できます。"),
            rx.hstack(
                rx.button("Refresh", on_click=HistoryState.refresh),
                rx.button("Load YAML Preview", on_click=HistoryState.load_yaml_preview),
                rx.button("Validate YAML", on_click=HistoryState.validate_yaml_editor),
                rx.button("Queue Rerun", on_click=HistoryState.rerun_selected),
                rx.button("Download YAML", on_click=HistoryState.download_yaml),
                spacing="3",
                wrap="wrap",
            ),
            rx.text(f"Source: {HistoryState.yaml_source_label}"),
            rx.input(
                placeholder="Output Name",
                value=HistoryState.selected_output_name,
                on_change=HistoryState.set_selected_output_name,
                width="100%",
            ),
            rx.hstack(
                rx.upload(
                    rx.button("Select YAML File"),
                    id=HISTORY_YAML_UPLOAD_ID,
                    accept={".yaml": [".yaml"], ".yml": [".yml"]},
                    max_files=1,
                ),
                rx.button(
                    "Import Upload",
                    on_click=HistoryState.handle_yaml_upload(
                        rx.upload_files(upload_id=HISTORY_YAML_UPLOAD_ID)
                    ),
                ),
                rx.text(rx.selected_files(HISTORY_YAML_UPLOAD_ID)),
                spacing="3",
                width="100%",
                wrap="wrap",
            ),
            rx.hstack(
                rx.input(
                    placeholder="Import YAML from server path",
                    value=HistoryState.import_path,
                    on_change=HistoryState.set_import_path,
                    width="100%",
                ),
                rx.button("Import Path", on_click=HistoryState.import_yaml_from_path),
                spacing="3",
                width="100%",
            ),
            rx.hstack(
                rx.input(
                    placeholder="Export YAML to server path",
                    value=HistoryState.export_path,
                    on_change=HistoryState.set_export_path,
                    width="100%",
                ),
                rx.button(
                    "Export Edited YAML",
                    on_click=HistoryState.export_editor_to_path,
                ),
                rx.button(
                    "Export History YAML",
                    on_click=HistoryState.export_selected_history_to_path,
                ),
                spacing="3",
                width="100%",
                wrap="wrap",
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
                min_height="12rem",
                width="100%",
            ),
            title="Recipe Preview",
        ),
        section_card(
            rx.text_area(
                value=HistoryState.yaml_editor_text,
                on_change=HistoryState.set_yaml_editor_text,
                min_height="24rem",
                width="100%",
            ),
            title="Editable YAML",
        ),
        current_route="/history",
        description="実行履歴を参照し、保存済み YAML の確認、編集、再投入を行います。",
        feedback_message=HistoryState.status_message,
        feedback_variant=HistoryState.status_variant,
        on_mount=HistoryState.load_page,
    )
