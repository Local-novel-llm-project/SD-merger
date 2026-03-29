from __future__ import annotations

from module.arthemy_tuner_config import ARTHEMY_TUNER_MODES, build_arthemy_tune_job_config
from ui.services.model_service import resolve_model_path
from ui.utils import enqueue_merge_task


def list_modes() -> list[str]:
    return list(ARTHEMY_TUNER_MODES)


def build_tuning_config(
    target_model: str,
    mode: str,
    clip_base_scale: float | None,
    unet_base_scale: float | None,
    vectors_override: str | None,
    output_name: str | None,
) -> tuple[dict, str]:
    target_model_path = resolve_model_path(target_model)
    if not target_model_path:
        raise ValueError("Target Model is required.")

    clip_overrides = {"base_scale": clip_base_scale} if clip_base_scale is not None else None
    unet_overrides = {"base_scale": unet_base_scale}
    if vectors_override:
        unet_overrides["vectors_override"] = vectors_override

    config = build_arthemy_tune_job_config(
        target_model=target_model_path,
        mode=mode,
        clip_overrides=clip_overrides,
        unet_overrides=unet_overrides,
        output_name=(output_name or "").strip() or None,
    )

    resolved_output_name = str(config.get("output_name") or "arthemy_tuned.safetensors")
    return config, resolved_output_name


def queue_tuning(config: dict, output_name: str) -> str:
    return enqueue_merge_task(
        config,
        output_name,
        task_name="Arthemy Tuning",
    )
