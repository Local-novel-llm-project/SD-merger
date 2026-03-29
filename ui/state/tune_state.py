from __future__ import annotations

import json

import reflex as rx

from ui.services.app_boot import ensure_app_ready
from ui.services.model_service import list_models
from ui.services.tuning_service import (
    build_tuning_config,
    build_tuning_preview,
    list_modes,
    parse_optional_float,
    queue_tuning,
)


class TuneState(rx.State):
    available_models: list[str] = []
    modes: list[str] = list_modes()

    target_model: str = ""
    mode: str = "Soft Value"
    clip_base_scale: str = "1.0"
    unet_base_scale: str = "1.0"
    vectors_override: str = ""
    output_name: str = "arthemy_tuned.safetensors"

    preview_json: str = "{}"
    status_message: str = ""
    status_variant: str = "info"
    last_task_id: str = ""

    def load_page(self) -> None:
        ensure_app_ready()
        self.available_models = list_models()
        self.refresh_preview()

    def set_target_model_value(self, value: str) -> None:
        self.target_model = value
        self.refresh_preview()

    def set_mode_value(self, value: str) -> None:
        self.mode = value
        self.refresh_preview()

    def set_clip_base_scale_value(self, value: str) -> None:
        self.clip_base_scale = value
        self.refresh_preview()

    def set_unet_base_scale_value(self, value: str) -> None:
        self.unet_base_scale = value
        self.refresh_preview()

    def set_vectors_override_value(self, value: str) -> None:
        self.vectors_override = value
        self.refresh_preview()

    def set_output_name_value(self, value: str) -> None:
        self.output_name = value
        self.refresh_preview()

    def refresh_preview(self) -> None:
        self.preview_json = build_tuning_preview(
            self.target_model,
            self.mode,
            self.clip_base_scale,
            self.unet_base_scale,
            self.vectors_override,
            self.output_name,
        )

    def queue_tuning_job(self) -> None:
        config, output_name = build_tuning_config(
            self.target_model,
            self.mode,
            parse_optional_float(self.clip_base_scale),
            parse_optional_float(self.unet_base_scale),
            self.vectors_override,
            self.output_name,
        )
        self.last_task_id = queue_tuning(config, output_name)
        self.status_message = f"Queued Arthemy Tuning task: {self.last_task_id}"
        self.status_variant = "success"
        self.preview_json = json.dumps(config, indent=2, ensure_ascii=False)
