from __future__ import annotations

import reflex as rx

from ui.pages.common import page_shell, section_card
from ui.state.tune_state import TuneState


def tune_page() -> rx.Component:
    return page_shell(
        "Arthemy Tuner",
        section_card(
            rx.text("Reflex UI では主要パラメータを services/state 経由で組み立ててキュー投入します。"),
            rx.hstack(
                rx.button("Refresh Preview", on_click=TuneState.refresh_preview),
                rx.button("Queue Tuning", on_click=TuneState.queue_tuning_job),
                spacing="3",
                wrap="wrap",
            ),
            rx.text(f"Last Task ID: {TuneState.last_task_id}"),
            title="Actions",
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
        ),
        section_card(
            rx.text_area(
                value=TuneState.preview_json,
                read_only=True,
                min_height="24rem",
                width="100%",
            ),
            title="Config Preview",
        ),
        current_route="/tune",
        description="Arthemy Tuner の主要パラメータを編集し、キュー投入前に設定内容を確認できます。",
        feedback_message=TuneState.status_message,
        feedback_variant=TuneState.status_variant,
        on_mount=TuneState.load_page,
    )
