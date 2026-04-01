from __future__ import annotations

import reflex as rx

from module.services.merge import (
    MERGE_STRATEGIES,
    MERGE_VELOCITY_HELP,
    TARGET_STRATEGIES,
    build_basic_merge_config,
    build_merge_preview,
    build_preview_json,
    create_default_output_name,
    queue_merge,
)
from ui.state.base import BasePageState


class MergeState(BasePageState):
    available_models: list[str] = []
    merge_strategies: list[str] = MERGE_STRATEGIES
    target_strategies: list[str] = TARGET_STRATEGIES
    velocity_help: str = MERGE_VELOCITY_HELP

    model_a: str = ""
    model_b: str = ""
    model_c: str = "選択しない"
    strategy: str = "mix"
    target_strategy: str = "mix"
    velocity: str = "0.5"
    left_right_velocity: str = ""
    mbw: str = ""
    bake_in_vae: str = ""
    output_name: str = ""
    lazy_load: bool = True
    use_advanced_options: bool = False

    status_message: str = "Models ディレクトリを読み込んでください。"
    status_variant: str = "info"
    preview_json: str = "{}"
    last_task_id: str = ""
    output_name_locked: bool = False

    def load_page(self) -> None:
        self.ensure_ready()
        self.refresh_models()
        self.refresh_preview()

    def _reconcile_model_selection(self) -> None:
        if self.model_a and self.model_a not in self.available_models:
            self.model_a = ""
        if self.model_b and self.model_b not in self.available_models:
            self.model_b = ""
        if self.model_c not in {"", "選択しない"} and self.model_c not in self.available_models:
            self.model_c = "選択しない"
        if self.bake_in_vae and self.bake_in_vae not in self.available_models:
            self.bake_in_vae = ""

    def set_model_a_value(self, value: str) -> None:
        self.model_a = value
        self._refresh_output_name()
        self.refresh_preview()

    def set_model_b_value(self, value: str) -> None:
        self.model_b = value
        self._refresh_output_name()
        self.refresh_preview()

    def set_model_c_value(self, value: str) -> None:
        self.model_c = value
        self.refresh_preview()

    def set_strategy_value(self, value: str) -> None:
        self.strategy = value
        self.refresh_preview()

    def set_target_strategy_value(self, value: str) -> None:
        self.target_strategy = value
        self.refresh_preview()

    def set_velocity_value(self, value: str) -> None:
        self.velocity = value
        self.refresh_preview()

    def set_left_right_velocity_value(self, value: str) -> None:
        self.left_right_velocity = value
        self.refresh_preview()

    def set_mbw_value(self, value: str) -> None:
        self.mbw = value
        self.refresh_preview()

    def set_bake_in_vae_value(self, value: str) -> None:
        self.bake_in_vae = value
        self.refresh_preview()

    def set_output_name_value(self, value: str) -> None:
        self.output_name = value
        self.output_name_locked = bool(value.strip())
        self.refresh_preview()

    def set_lazy_load_value(self, value: bool) -> None:
        self.lazy_load = bool(value)
        self.refresh_preview()

    def set_use_advanced_options_value(self, value: bool) -> None:
        self.use_advanced_options = bool(value)
        self.refresh_preview()

    def _refresh_output_name(self) -> None:
        if self.output_name_locked:
            return
        if self.model_a and self.model_b:
            self.output_name = create_default_output_name(self.model_a, self.model_b)

    def refresh_preview(self) -> None:
        self.preview_json = build_merge_preview(
            self.model_a,
            self.model_b,
            self.model_c,
            self.strategy,
            self.target_strategy,
            self.velocity,
            self.left_right_velocity,
            self.use_advanced_options,
            self.mbw,
            self.bake_in_vae,
            self.output_name,
            self.lazy_load,
        )

    def queue_current_merge(self) -> None:
        self.begin_busy("マージ設定をキューに追加しています。")
        try:
            config, output_name = build_basic_merge_config(
                self.model_a,
                self.model_b,
                self.model_c,
                self.strategy,
                self.target_strategy,
                float(self.velocity),
                self.left_right_velocity,
                self.use_advanced_options,
                self.mbw,
                self.bake_in_vae,
                self.output_name,
                self.lazy_load,
            )
            self.last_task_id = queue_merge(
                config,
                output_name,
                task_name=f"Merge: {self.strategy}",
            )
            self.preview_json = build_preview_json(config)
            self.end_busy(f"Queued merge task: {self.last_task_id}")
        except Exception as exc:
            self.fail_busy(exc, action="マージのキュー投入")
