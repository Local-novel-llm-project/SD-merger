from __future__ import annotations

import reflex as rx

from ui.pages.common import log_panel, page_shell, section_card
from ui.state.merge_state import MergeState


def merge_page() -> rx.Component:
    return page_shell(
        "SD-merger",
        section_card(
            rx.text("Reflex ベースの Merge Models 画面です。主要な設定を queue ベースで実行します。"),
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
                    "Merge Models",
                    on_click=MergeState.queue_current_merge,
                    disabled=MergeState.busy,
                    loading=MergeState.busy,
                ),
                spacing="3",
                wrap="wrap",
            ),
            title="Actions",
            description="モデル一覧更新、プレビュー更新、キュー投入の主要操作をここに集約します。",
        ),
        section_card(
            rx.vstack(
                rx.text("Model A (Left)"),
                rx.select(
                    MergeState.available_models,
                    value=MergeState.model_a,
                    on_change=MergeState.set_model_a_value,
                    placeholder="Select Model A",
                    width="100%",
                ),
                rx.text("Model B (Right)"),
                rx.select(
                    MergeState.available_models,
                    value=MergeState.model_b,
                    on_change=MergeState.set_model_b_value,
                    placeholder="Select Model B",
                    width="100%",
                ),
                rx.text("Model C (Base/Target, optional)"),
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
                rx.text("Velocity (Target / Final)"),
                rx.input(
                    value=MergeState.velocity,
                    on_change=MergeState.set_velocity_value,
                    width="100%",
                ),
                rx.text(MergeState.velocity_help, color="#64748b"),
                width="100%",
                spacing="3",
                align="start",
            ),
            title="Merge Settings",
            description="マージ対象、戦略、速度パラメータを旧タブ相当の構成で編集します。",
        ),
        section_card(
            rx.vstack(
                rx.checkbox(
                    "Enable Advanced Options",
                    checked=MergeState.use_advanced_options,
                    on_change=MergeState.set_use_advanced_options_value,
                ),
                rx.cond(
                    MergeState.use_advanced_options,
                    rx.vstack(
                        rx.text("Merge Block Weight (MBW)"),
                        rx.text_area(
                            value=MergeState.mbw,
                            on_change=MergeState.set_mbw_value,
                            min_height="8rem",
                            width="100%",
                            placeholder="e.g. 1,0.5,0.5,0...",
                        ),
                        rx.text("LRV (A/B Strategy, optional)"),
                        rx.input(
                            value=MergeState.left_right_velocity,
                            on_change=MergeState.set_left_right_velocity_value,
                            width="100%",
                            placeholder="blank = auto (AB uses Velocity, ABC uses 1.0)",
                        ),
                        rx.text("Bake in VAE"),
                        rx.select(
                            [""] + MergeState.available_models,
                            value=MergeState.bake_in_vae,
                            on_change=MergeState.set_bake_in_vae_value,
                            width="100%",
                            placeholder="Select VAE",
                        ),
                        rx.text("Output Filename"),
                        rx.input(
                            value=MergeState.output_name,
                            on_change=MergeState.set_output_name_value,
                            width="100%",
                        ),
                        rx.checkbox(
                            "Enable Lazy Load (Memory saving)",
                            checked=MergeState.lazy_load,
                            on_change=MergeState.set_lazy_load_value,
                        ),
                        width="100%",
                        spacing="3",
                        align="start",
                    ),
                    rx.text(
                        "MBW / LRV / Bake in VAE / Output Filename / Lazy Load を必要なときだけ展開します。",
                        color="#64748b",
                    ),
                ),
                width="100%",
                spacing="3",
                align="start",
            ),
            title="Advanced Options",
            description="旧 Merge Models タブの詳細設定を Reflex から編集します。",
        ),
        section_card(
            log_panel(MergeState.preview_json, min_height="24rem"),
            title="Config Preview",
            description="実際にキューへ送る設定 JSON を確認します。",
        ),
        current_route="/",
        description="旧 Merge Models タブ相当の設定を Reflex 上で編集し、生成される設定のプレビューを確認できます。",
        feedback_message=MergeState.status_message,
        feedback_variant=MergeState.status_variant,
        busy_message=MergeState.busy_message,
        header_actions=rx.vstack(
            rx.text(f"Last Task ID: {MergeState.last_task_id}", color="#6a5b4d", size="2"),
            rx.text("Merge Models workflow", color="#8b7a68", size="2"),
            spacing="1",
            align="end",
        ),
        on_mount=MergeState.load_page,
    )
