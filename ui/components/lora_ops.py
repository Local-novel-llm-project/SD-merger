import os

import gradio as gr

from module.error_messages import build_user_error_message
from module.lora_input import (
    is_compact_lora_spec_text,
    normalize_lora_models_and_ratios,
)
from ui.utils import enqueue_merge_task, get_model_list, get_model_path


DEFAULT_LORA_RATIO = 1.0
REQUIRED_LORA_MODELS_MESSAGE = "At least one LoRA model is required."
INVALID_RATIO_FORMAT_MESSAGE = (
    "Invalid format. Use '0.5, 1.0' or "
    "'lora_a.safetensors:0.5, lora_b.safetensors:1.0'."
)


def _resolve_output_path(output_name):
    if os.path.isabs(output_name):
        return os.path.abspath(output_name)
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "output", output_name
        )
    )


def _normalize_selected_lora_models(selected_models):
    if not selected_models:
        return []

    return [
        model_name
        for model_name in (str(model).strip() for model in selected_models)
        if model_name
    ]


def _coerce_lora_ratio_value(value, default_ratio=DEFAULT_LORA_RATIO):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default_ratio


def _sync_lora_ratio_map(
    selected_models,
    ratio_map,
    *,
    default_ratio=DEFAULT_LORA_RATIO,
):
    normalized_models = _normalize_selected_lora_models(selected_models)
    current_ratio_map = dict(ratio_map or {})
    return {
        model_name: _coerce_lora_ratio_value(
            current_ratio_map.get(model_name, default_ratio),
            default_ratio,
        )
        for model_name in normalized_models
    }


def _update_lora_ratio_value(value, ratio_map, model_name):
    updated_ratio_map = dict(ratio_map or {})
    updated_ratio_map[model_name] = _coerce_lora_ratio_value(value)
    return updated_ratio_map


def _parse_compact_lora_ratio_text(ratio_text):
    try:
        if not is_compact_lora_spec_text(ratio_text):
            raise ValueError("Invalid compact LoRA ratio format.")
        model_names, ratio_list = normalize_lora_models_and_ratios(ratio_text)
    except ValueError:
        return None, None, INVALID_RATIO_FORMAT_MESSAGE

    if not model_names:
        return None, None, REQUIRED_LORA_MODELS_MESSAGE

    return model_names, dict(zip(model_names, ratio_list)), None


def _import_compact_lora_ratio_text(ratio_text, current_models, current_ratio_map):
    model_names, ratio_map, error = _parse_compact_lora_ratio_text(ratio_text)
    if error:
        return (
            _normalize_selected_lora_models(current_models),
            dict(current_ratio_map or {}),
            error,
        )

    return model_names, ratio_map, f"Imported {len(model_names)} LoRA ratio(s)."


def _resolve_lora_models_and_ratios(selected_models, ratio_map):
    model_names = _normalize_selected_lora_models(selected_models)
    if not model_names:
        return None, None, REQUIRED_LORA_MODELS_MESSAGE

    synced_ratio_map = _sync_lora_ratio_map(model_names, ratio_map)
    return (
        [get_model_path(model_name) for model_name in model_names],
        [synced_ratio_map[model_name] for model_name in model_names],
        None,
    )


def _render_lora_ratio_selector(key_prefix):
    models = gr.Dropdown(
        label="LoRA Models",
        choices=get_model_list(),
        multiselect=True,
    )
    ratio_state = gr.State({})

    models.change(
        _sync_lora_ratio_map,
        inputs=[models, ratio_state],
        outputs=[ratio_state],
        queue=False,
    )

    @gr.render(inputs=[models, ratio_state], queue=False, show_progress="hidden")
    def render_ratio_inputs(selected_models, current_ratio_map):
        synced_ratio_map = _sync_lora_ratio_map(selected_models, current_ratio_map)

        if not synced_ratio_map:
            gr.Markdown("Select one or more LoRA models to edit ratios.")
            return

        gr.Markdown("### LoRA Ratios")
        for model_name, ratio_value in synced_ratio_map.items():
            model_key = gr.State(model_name)
            ratio_input = gr.Number(
                label=model_name,
                value=ratio_value,
                step=0.01,
                key=(key_prefix, model_name),
            )
            ratio_input.change(
                _update_lora_ratio_value,
                inputs=[ratio_input, ratio_state, model_key],
                outputs=[ratio_state],
                queue=False,
            )

    with gr.Accordion("Advanced Ratio Import", open=False):
        compact_ratio_text = gr.Textbox(
            label="Compact LoRA:ratio list",
            placeholder="style_a.safetensors:0.4, style_b.safetensors:0.9",
        )
        import_ratios_btn = gr.Button("Import Compact Spec")
        import_status = gr.Textbox(label="Import Status", value="", interactive=False)
        import_ratios_btn.click(
            _import_compact_lora_ratio_text,
            inputs=[compact_ratio_text, models, ratio_state],
            outputs=[models, ratio_state, import_status],
            queue=False,
        )

    return models, ratio_state


def render_lora_ops_tab():
    with gr.Tabs():
        with gr.TabItem("Extract LoRA"):
            gr.Markdown("### Extract LoRA from Checkpoints")
            with gr.Row():
                with gr.Column(scale=1):
                    model_list = get_model_list()
                    base_model = gr.Dropdown(
                        label="Base Model (Original)",
                        choices=model_list,
                    )
                    tuned_model = gr.Dropdown(
                        label="Tuned Model (Finetuned)",
                        choices=model_list,
                    )
                    extract_output = gr.Textbox(
                        label="Output Filename",
                        value="extracted_lora.safetensors",
                    )

                with gr.Column(scale=1):
                    dim = gr.Number(label="Network Dim (Rank)", value=128, precision=0)
                    conv_dim = gr.Number(label="Conv Dim", value=0, precision=0)
                    alpha = gr.Number(label="Alpha", value=1.0)
                    t_strategy = gr.Dropdown(
                        label="Target Strategy",
                        choices=[
                            "mix",
                            "addition",
                            "subtraction",
                            "angle",
                            "trainDifference",
                            "extract",
                        ],
                        value="extract",
                    )
                    device = gr.Dropdown(
                        label="Device", choices=["cpu", "cuda"], value="cpu"
                    )
                    save_precision = gr.Dropdown(
                        label="Save Precision",
                        choices=["float", "fp16", "bf16"],
                        value="float",
                    )

                    with gr.Row():
                        is_sdxl_ext = gr.Checkbox(label="Is SDXL", value=False)
                        is_v2_ext = gr.Checkbox(label="Is V2", value=False)

            extract_btn = gr.Button("Extract LoRA", variant="primary")
            extract_log = gr.Textbox(label="Extraction Log")

            def run_extract(base, tuned, out, d, cd, a, t_strat, dev, prec, sdxl, v2):
                if not base or not tuned:
                    return "Base Model and Tuned Model are required."

                config = {
                    "lora_ops": {
                        "stop_after_lora_ops": True,
                        "operations": [
                            {
                                "type": "extract",
                                "base_model": get_model_path(base),
                                "tuned_model": get_model_path(tuned),
                                "output": _resolve_output_path(out),
                                "dim": int(d),
                                "conv_dim": int(cd) if cd > 0 else None,
                                "alpha": float(a),
                                "target_strategy": t_strat,
                                "device": dev,
                                "save_precision": prec,
                                "sdxl": sdxl,
                                "v2": v2,
                            }
                        ],
                    }
                }
                return _run_lora_config(config, out, "Extract")

            extract_btn.click(
                run_extract,
                inputs=[
                    base_model,
                    tuned_model,
                    extract_output,
                    dim,
                    conv_dim,
                    alpha,
                    t_strategy,
                    device,
                    save_precision,
                    is_sdxl_ext,
                    is_v2_ext,
                ],
                outputs=[extract_log],
            )

        with gr.TabItem("Merge LoRAs"):
            gr.Markdown("### Merge Multiple LoRAs")
            with gr.Row():
                with gr.Column(scale=1):
                    models, merge_ratio_state = _render_lora_ratio_selector("merge")
                    strategy = gr.Dropdown(
                        label="Merge Strategy",
                        choices=[
                            "addition",
                            "subtraction",
                            "multiplication",
                            "mix",
                            "cosineA",
                            "cosineB",
                            "smoothAdd",
                            "tensor",
                            "tensor2",
                            "mbw_each",
                            "quantum",
                        ],
                        value="addition",
                    )
                    merge_output = gr.Textbox(
                        label="Output Filename",
                        value="merged_lora.safetensors",
                    )

                with gr.Column(scale=1):
                    precision = gr.Dropdown(
                        label="Compute Precision",
                        choices=["float", "fp16", "bf16"],
                        value="float",
                    )
                    m_save_precision = gr.Dropdown(
                        label="Save Precision",
                        choices=["float", "fp16", "bf16"],
                        value="float",
                    )
                    concat = gr.Checkbox(label="Concat", value=False)
                    shuffle = gr.Checkbox(label="Shuffle", value=False)

                    with gr.Row():
                        is_sdxl_mrg = gr.Checkbox(label="Is SDXL", value=False)
                        is_v2_mrg = gr.Checkbox(label="Is V2", value=False)

            merge_btn = gr.Button("Merge LoRAs", variant="primary")
            merge_log = gr.Textbox(label="Merge Log")

            def run_merge(
                mods,
                ratio_map,
                strat,
                out,
                prec,
                s_prec,
                conc,
                shuf,
                sdxl,
                v2,
            ):
                model_paths, ratio_list, error = _resolve_lora_models_and_ratios(
                    mods,
                    ratio_map,
                )
                if error:
                    return error

                config = {
                    "lora_ops": {
                        "stop_after_lora_ops": True,
                        "operations": [
                            {
                                "type": "merge",
                                "models": model_paths,
                                "ratios": ratio_list,
                                "strategy": strat,
                                "output": _resolve_output_path(out),
                                "precision": prec,
                                "save_precision": s_prec,
                                "concat": conc,
                                "shuffle": shuf,
                                "sdxl": sdxl,
                                "v2": v2,
                            }
                        ],
                    }
                }
                return _run_lora_config(config, out, "Merge")

            merge_btn.click(
                run_merge,
                inputs=[
                    models,
                    merge_ratio_state,
                    strategy,
                    merge_output,
                    precision,
                    m_save_precision,
                    concat,
                    shuffle,
                    is_sdxl_mrg,
                    is_v2_mrg,
                ],
                outputs=[merge_log],
            )

        with gr.TabItem("Merge LoRA into Model"):
            gr.Markdown("### Merge One or More LoRAs into a Checkpoint")
            with gr.Row():
                with gr.Column(scale=1):
                    base_checkpoint = gr.Dropdown(
                        label="Base Checkpoint",
                        choices=get_model_list(),
                    )
                    checkpoint_loras, checkpoint_ratio_state = _render_lora_ratio_selector(
                        "apply"
                    )
                    checkpoint_output = gr.Textbox(
                        label="Output Filename",
                        value="model_with_lora.safetensors",
                    )

                with gr.Column(scale=1):
                    checkpoint_precision = gr.Dropdown(
                        label="Compute Precision",
                        choices=["float", "fp16", "bf16"],
                        value="float",
                    )
                    checkpoint_save_precision = gr.Dropdown(
                        label="Save Precision",
                        choices=["float", "fp16", "bf16"],
                        value="float",
                    )

                    with gr.Row():
                        is_sdxl_apply = gr.Checkbox(label="Is SDXL", value=False)
                        is_v2_apply = gr.Checkbox(label="Is V2", value=False)

            apply_btn = gr.Button("Merge LoRA into Model", variant="primary")
            apply_log = gr.Textbox(label="Apply Log")

            def run_apply(base, mods, ratio_map, out, prec, s_prec, sdxl, v2):
                if not base:
                    return "Base checkpoint is required."
                model_paths, ratio_list, error = _resolve_lora_models_and_ratios(
                    mods,
                    ratio_map,
                )
                if error:
                    return error

                config = {
                    "lora_ops": {
                        "stop_after_lora_ops": True,
                        "operations": [
                            {
                                "type": "apply",
                                "sd_model": get_model_path(base),
                                "models": model_paths,
                                "ratios": ratio_list,
                                "output": _resolve_output_path(out),
                                "precision": prec,
                                "save_precision": s_prec,
                                "sdxl": sdxl,
                                "v2": v2,
                            }
                        ],
                    }
                }
                return _run_lora_config(config, out, "Apply")

            apply_btn.click(
                run_apply,
                inputs=[
                    base_checkpoint,
                    checkpoint_loras,
                    checkpoint_ratio_state,
                    checkpoint_output,
                    checkpoint_precision,
                    checkpoint_save_precision,
                    is_sdxl_apply,
                    is_v2_apply,
                ],
                outputs=[apply_log],
            )


def _run_lora_config(config, out_name, op_name):
    try:
        task_id = enqueue_merge_task(config, out_name, task_name=f"LoRA {op_name}")
        return (
            f"LoRA {op_name} task '{task_id}' added to queue. Output will be {out_name}"
        )
    except Exception as e:
        return build_user_error_message(e, action=f"LoRA {op_name} タスクの追加")
