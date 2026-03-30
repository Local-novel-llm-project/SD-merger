from __future__ import annotations

import reflex as rx

from ui.pages.common import log_panel, meta_badge, page_shell, record_row, section_card
from ui.state.history_state import HISTORY_YAML_UPLOAD_ID, HistoryState


def _history_row(row: dict[str, str]) -> rx.Component:
    return record_row(
        row["output_name"],
        status=row["status"],
        subtitle=f"{row['model_a']}  <-  {row['model_b']}",
        meta=[
            meta_badge(row["date"]),
            meta_badge(f"Strategy: {row['strategy']}"),
            meta_badge(f"Velocity: {row['velocity']} / LRV: {row['left_right_velocity']}"),
        ],
        action=rx.button(
            "Load",
            size="1",
            variant="soft",
            on_click=HistoryState.select_history_entry(row["output_name"]),
        ),
    )


def history_page() -> rx.Component:
    return page_shell(
        "History",
        section_card(
            rx.text("過去の実行履歴を確認し、YAML を編集・import/export して再実行できます。"),
            rx.hstack(
                rx.button("Refresh", on_click=HistoryState.refresh, disabled=HistoryState.busy),
                rx.button(
                    "Load YAML Preview",
                    on_click=HistoryState.load_yaml_preview,
                    disabled=HistoryState.busy,
                    loading=HistoryState.busy,
                ),
                rx.button(
                    "Validate YAML",
                    on_click=HistoryState.validate_yaml_editor,
                    disabled=HistoryState.busy,
                    loading=HistoryState.busy,
                ),
                rx.button(
                    "Queue Rerun",
                    on_click=HistoryState.rerun_selected,
                    disabled=HistoryState.busy,
                    loading=HistoryState.busy,
                ),
                rx.button(
                    "Download YAML",
                    on_click=HistoryState.download_yaml,
                    disabled=HistoryState.busy,
                ),
                spacing="3",
                wrap="wrap",
            ),
            rx.text(f"Source: {HistoryState.yaml_source_label}"),
            rx.input(
                placeholder="Output Name",
                value=HistoryState.selected_output_name,
                on_change=HistoryState.set_selected_output_name,
                disabled=HistoryState.busy,
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
                    disabled=HistoryState.busy,
                    loading=HistoryState.busy,
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
                    disabled=HistoryState.busy,
                    width="100%",
                ),
                rx.button(
                    "Import Path",
                    on_click=HistoryState.import_yaml_from_path,
                    disabled=HistoryState.busy,
                    loading=HistoryState.busy,
                ),
                spacing="3",
                width="100%",
            ),
            rx.hstack(
                rx.input(
                    placeholder="Export YAML to server path",
                    value=HistoryState.export_path,
                    on_change=HistoryState.set_export_path,
                    disabled=HistoryState.busy,
                    width="100%",
                ),
                rx.button(
                    "Export Edited YAML",
                    on_click=HistoryState.export_editor_to_path,
                    disabled=HistoryState.busy,
                    loading=HistoryState.busy,
                ),
                rx.button(
                    "Export History YAML",
                    on_click=HistoryState.export_selected_history_to_path,
                    disabled=HistoryState.busy,
                    loading=HistoryState.busy,
                ),
                spacing="3",
                width="100%",
                wrap="wrap",
            ),
            title="History Controls",
            description="履歴の選択、YAML の import/export、再投入操作をまとめています。",
        ),
        section_card(
            rx.vstack(
                rx.foreach(HistoryState.history_rows, _history_row),
                width="100%",
                align="start",
                spacing="3",
            ),
            title="History Entries",
            description="保存済みレシピを状態付きで一覧表示します。",
        ),
        section_card(
            log_panel(HistoryState.yaml_preview, min_height="12rem"),
            title="Recipe Preview",
            description="選択中レシピの読み取り専用プレビューです。",
        ),
        section_card(
            rx.box(
                log_panel(
                    HistoryState.yaml_editor_text,
                    min_height="24rem",
                    read_only=False,
                    on_change=HistoryState.set_yaml_editor_text,
                ),
                width="100%",
                opacity=rx.cond(HistoryState.busy, "0.75", "1"),
            ),
            title="Editable YAML",
            description="履歴や import から読み込んだ YAML を直接編集します。",
        ),
        current_route="/history",
        description="実行履歴を参照し、保存済み YAML の確認、編集、再投入を行います。",
        feedback_message=HistoryState.status_message,
        feedback_variant=HistoryState.status_variant,
        busy_message=HistoryState.busy_message,
        header_actions=rx.vstack(
            rx.text(
                f"Source: {HistoryState.yaml_source_label}",
                color="#6a5b4d",
                size="2",
            ),
            rx.text(
                f"Selected: {HistoryState.selected_output_name}",
                color="#6a5b4d",
                size="2",
            ),
            spacing="1",
            align="end",
        ),
        on_mount=HistoryState.load_page,
    )
