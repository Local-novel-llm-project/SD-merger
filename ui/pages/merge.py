from __future__ import annotations

import reflex as rx

from ui.pages.common import page_shell, section_card
from ui.state.merge_state import MergeState


def merge_page() -> rx.Component:
    return page_shell(
        "SD-merger",
        section_card(
            rx.text("Reflex ベースのメインマージ画面です。主要な設定を queue ベースで実行します。"),
            rx.hstack(
                rx.button(
                    "Refresh Models",
                    on_click=MergeState.refresh_models,
                    disabled=MergeState.busy,
                    loading=MergeState.busy,
                ),
                rx.button(
                    "Refresh Preview",
                    on_click=MergeState.refresh_preview,
                    disabled=MergeState.busy,
                ),
                rx.button(
                    "Queue Merge",
                    on_click=MergeState.queue_current_merge,
                    disabled=MergeState.busy,
                    loading=MergeState.busy,
                ),
                spacing="3",
                wrap="wrap",
            ),
            rx.text(f"Last Task ID: {MergeState.last_task_id}"),
            rx.cond(MergeState.busy_message != "", rx.text(MergeState.busy_message)),
            title="Actions",
        ),
        section_card(
            rx.vstack(
                rx.text("Model A"),
                rx.select(
                    MergeState.available_models,
                    value=MergeState.model_a,
                    on_change=MergeState.set_model_a_value,
                    placeholder="Select Model A",
                    width="100%",
                ),
                rx.text("Model B"),
                rx.select(
                    MergeState.available_models,
                    value=MergeState.model_b,
                    on_change=MergeState.set_model_b_value,
                    placeholder="Select Model B",
                    width="100%",
                ),
                rx.text("Model C / Base Target"),
                rx.select(
                    ["選択しない"] + MergeState.available_models,
                    value=MergeState.model_c,
                    on_change=MergeState.set_model_c_value,
                    width="100%",
                ),
                rx.text("Strategy"),
                rx.select(
                    MergeState.merge_strategies,
                    value=MergeState.strategy,
                    on_change=MergeState.set_strategy_value,
                    width="100%",
                ),
                rx.text("Target Strategy"),
                rx.select(
                    MergeState.target_strategies,
                    value=MergeState.target_strategy,
                    on_change=MergeState.set_target_strategy_value,
                    width="100%",
                ),
                rx.text("Velocity"),
                rx.input(
                    value=MergeState.velocity,
                    on_change=MergeState.set_velocity_value,
                    width="100%",
                ),
                rx.text("A/B Strategy Velocity"),
                rx.input(
                    value=MergeState.left_right_velocity,
                    on_change=MergeState.set_left_right_velocity_value,
                    width="100%",
                ),
                rx.text("Output Name"),
                rx.input(
                    value=MergeState.output_name,
                    on_change=MergeState.set_output_name_value,
                    width="100%",
                ),
                rx.checkbox(
                    "Use Advanced Options",
                    checked=MergeState.use_advanced_options,
                    on_change=MergeState.set_use_advanced_options_value,
                ),
                rx.checkbox(
                    "Lazy Load",
                    checked=MergeState.lazy_load,
                    on_change=MergeState.set_lazy_load_value,
                ),
                rx.text("MBW"),
                rx.text_area(
                    value=MergeState.mbw,
                    on_change=MergeState.set_mbw_value,
                    min_height="8rem",
                    width="100%",
                ),
                rx.text("Bake in VAE"),
                rx.select(
                    [""] + MergeState.available_models,
                    value=MergeState.bake_in_vae,
                    on_change=MergeState.set_bake_in_vae_value,
                    width="100%",
                ),
                rx.text(MergeState.velocity_help, color="#64748b"),
                width="100%",
                spacing="3",
                align="start",
            ),
            title="Merge Settings",
        ),
        section_card(
            rx.text_area(
                value=MergeState.preview_json,
                read_only=True,
                min_height="24rem",
                width="100%",
            ),
            title="Config Preview",
        ),
        current_route="/",
        description="モデル選択、速度パラメータ、出力名をまとめて管理し、生成される設定のプレビューを確認できます。",
        feedback_message=MergeState.status_message,
        feedback_variant=MergeState.status_variant,
        on_mount=MergeState.load_page,
    )
