from __future__ import annotations

import reflex as rx

from ui.pages.common import log_panel, page_shell, section_card
from ui.state.tune_state import TuneState


def tune_page() -> rx.Component:
    return page_shell(
        "Arthemy Tuner",
        section_card(
            rx.text("Reflex UI では主要パラメータを services/state 経由で組み立ててキュー投入します。"),
            rx.hstack(
                rx.button(
                    "Refresh Preview",
                    on_click=TuneState.refresh_preview,
                    disabled=TuneState.busy,
                ),
                rx.button(
                    "Queue Tuning",
                    on_click=TuneState.queue_tuning_job,
                    disabled=TuneState.busy,
                    loading=TuneState.busy,
                ),
                spacing="3",
                wrap="wrap",
            ),
            title="Actions",
            description="プレビュー更新とチューニングジョブ投入の主要操作をここに集約します。",
        ),
        section_card(
            rx.vstack(
                rx.text("Target Model"),
                rx.select(
                    TuneState.available_models,
                    value=TuneState.target_model,
                    on_change=TuneState.set_target_model_value,
                    placeholder="Select Target Model",
                    width="100%",
                ),
                rx.text("Mode"),
                rx.select(
                    TuneState.modes,
                    value=TuneState.mode,
                    on_change=TuneState.set_mode_value,
                    width="100%",
                ),
                rx.text("CLIP Base Scale"),
                rx.input(
                    value=TuneState.clip_base_scale,
                    on_change=TuneState.set_clip_base_scale_value,
                    width="100%",
                ),
                rx.text("UNet Base Scale"),
                rx.input(
                    value=TuneState.unet_base_scale,
                    on_change=TuneState.set_unet_base_scale_value,
                    width="100%",
                ),
                rx.text("Vectors Override"),
                rx.text_area(
                    value=TuneState.vectors_override,
                    on_change=TuneState.set_vectors_override_value,
                    min_height="8rem",
                    width="100%",
                ),
                rx.text("Output Name"),
                rx.input(
                    value=TuneState.output_name,
                    on_change=TuneState.set_output_name_value,
                    width="100%",
                ),
                width="100%",
                spacing="3",
                align="start",
            ),
            title="Tuning Settings",
            description="対象モデル、モード、ベーススケール、ベクトル上書きを編集します。",
        ),
        section_card(
            log_panel(TuneState.preview_json, min_height="24rem"),
            title="Config Preview",
            description="チューニング設定の最終 JSON を確認します。",
        ),
        current_route="/tune",
        description="Arthemy Tuner の主要パラメータを編集し、キュー投入前に設定内容を確認できます。",
        feedback_message=TuneState.status_message,
        feedback_variant=TuneState.status_variant,
        busy_message=TuneState.busy_message,
        header_actions=rx.text(
            f"Last Task ID: {TuneState.last_task_id}",
            color="#6a5b4d",
            size="2",
        ),
        on_mount=TuneState.load_page,
    )
