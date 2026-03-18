import os

import gradio as gr

from module.arthemy_tuner_config import (
    ARTHEMY_TUNER_MODES,
    CLIP_FIELD_SPECS,
    UNET_SECTION_SPECS,
    build_arthemy_tune_job_config,
)
from module.error_messages import build_user_error_message
from module.utility import generate_filename
from ui.utils import enqueue_merge_task, get_model_list, get_model_path


def _create_tuning_output_name(model_name: str) -> str:
    stem = os.path.splitext(os.path.basename(model_name))[0]
    return generate_filename(stem, "arthemy_tuned")


def _build_clip_overrides(values: list[float]) -> dict[str, float]:
    return {
        spec["name"]: float(value) for spec, value in zip(CLIP_FIELD_SPECS, values)
    }


def _build_unet_overrides(vectors_override: str, values: list[float]) -> dict[str, object]:
    overrides: dict[str, object] = {"base_scale": float(values[0])}

    field_specs = [
        field
        for section in UNET_SECTION_SPECS
        for field in section["fields"]
    ]
    for spec, value in zip(field_specs, values[1:]):
        overrides[spec["name"]] = float(value)

    if vectors_override.strip():
        overrides["vectors_override"] = vectors_override
    return overrides


def _queue_arthemy_tuning_task(
    model_name: str,
    mode: str,
    output_name: str,
    vectors_override: str,
    *values: float,
) -> str:
    if not model_name:
        return "Target model is required."

    clip_value_count = len(CLIP_FIELD_SPECS)
    clip_values = list(values[:clip_value_count])
    unet_values = list(values[clip_value_count:])

    try:
        config = build_arthemy_tune_job_config(
            target_model=get_model_path(model_name),
            mode=mode,
            clip_overrides=_build_clip_overrides(clip_values),
            unet_overrides=_build_unet_overrides(vectors_override, unet_values),
            output_name=output_name.strip() or None,
        )
        resolved_output_name = output_name.strip() or _create_tuning_output_name(
            model_name
        )
        task_id = enqueue_merge_task(
            config,
            resolved_output_name,
            task_name="Arthemy Tuner",
        )
    except Exception as exc:
        return build_user_error_message(exc, action="Arthemy Tuner タスクの追加")

    return (
        f"Arthemy tuning task '{task_id}' added to queue. "
        f"Output will be {resolved_output_name}"
    )


def _refresh_model_dropdown():
    return gr.update(choices=get_model_list())


def _render_slider(spec: dict, minimum: float = 0.0, maximum: float = 2.0):
    return gr.Slider(
        label=spec["label"],
        info=spec["description"],
        minimum=minimum,
        maximum=maximum,
        step=0.01,
        value=float(spec["default"]),
    )


def render_arthemy_tuner_tab():
    gr.Markdown("### Arthemy Tuner")
    gr.Markdown(
        "単一モデルに対して CLIP / U-Net の重みを live tuning するタスクをキューへ登録します。"
    )

    with gr.Row():
        refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
        target_model = gr.Dropdown(label="Target Model", choices=get_model_list())
        mode = gr.Dropdown(
            label="Mode",
            choices=ARTHEMY_TUNER_MODES,
            value=ARTHEMY_TUNER_MODES[0],
        )
        output_name = gr.Textbox(
            label="Output Filename",
            placeholder="Optional. Leave blank for auto naming.",
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### CLIP")
            clip_components = [_render_slider(spec) for spec in CLIP_FIELD_SPECS]

        with gr.Column(scale=1):
            gr.Markdown("#### U-Net")
            unet_base_scale = gr.Slider(
                label="U-Net Base Scale",
                info="U-Net 全体へ適用する基本倍率。",
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                value=1.0,
            )
            vectors_override = gr.Textbox(
                label="Vectors Override",
                placeholder="Optional: 19 comma-separated values",
            )

            unet_components = [unet_base_scale]
            for section in UNET_SECTION_SPECS:
                with gr.Accordion(section["title"], open=True):
                    for field in section["fields"]:
                        slider = _render_slider(field)
                        unet_components.append(slider)

    queue_btn = gr.Button("Queue Arthemy Tuning", variant="primary")
    output_log = gr.Textbox(label="Output Log")

    refresh_models_btn.click(
        _refresh_model_dropdown,
        inputs=[],
        outputs=[target_model],
    )
    queue_btn.click(
        _queue_arthemy_tuning_task,
        inputs=[target_model, mode, output_name, vectors_override]
        + clip_components
        + unet_components,
        outputs=[output_log],
    )
