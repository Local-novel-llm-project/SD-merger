import os
import gradio as gr
from ui.utils import get_model_list, get_model_path


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
                    extract_output = gr.Textbox(label="Output Filename", value="extracted_lora.safetensors")

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
                    device = gr.Dropdown(label="Device", choices=["cpu", "cuda"], value="cpu")
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
                                "output": (
                                    os.path.abspath(
                                        os.path.join(os.path.dirname(__file__), "..", "..", "models", "output", out)
                                    )
                                    if not os.path.isabs(out)
                                    else os.path.abspath(out)
                                ),
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
                    models = gr.Dropdown(
                        label="LoRA Models",
                        choices=get_model_list(),
                        multiselect=True,
                    )
                    ratios = gr.Textbox(
                        label="Ratios (comma separated)",
                        value="1.0, 1.0",
                        placeholder="1.0, 0.5",
                    )
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
                    merge_output = gr.Textbox(label="Output Filename", value="merged_lora.safetensors")

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

            def run_merge(mods, rats, strat, out, prec, s_prec, conc, shuf, sdxl, v2):
                if not mods or len(mods) < 1:
                    return "At least one LoRA model is required."

                try:
                    ratio_list = [float(r.strip()) for r in rats.split(",")]
                except ValueError:
                    return "Invalid ratios format. Must be comma separated numbers."

                model_paths = [get_model_path(m) for m in mods]

                if len(model_paths) != len(ratio_list):
                    # Pad ratios with 1.0 if not enough provided
                    if len(ratio_list) < len(model_paths):
                        ratio_list.extend([1.0] * (len(model_paths) - len(ratio_list)))
                    else:
                        ratio_list = ratio_list[: len(model_paths)]

                config = {
                    "lora_ops": {
                        "stop_after_lora_ops": True,
                        "operations": [
                            {
                                "type": "merge",
                                "models": model_paths,
                                "ratios": ratio_list,
                                "strategy": strat,
                                "output": (
                                    os.path.abspath(
                                        os.path.join(os.path.dirname(__file__), "..", "..", "models", "output", out)
                                    )
                                    if not os.path.isabs(out)
                                    else os.path.abspath(out)
                                ),
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
                    ratios,
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


def _run_lora_config(config, out_name, op_name):
    from module.queue_manager import queue_manager

    try:
        task_id = queue_manager.add_task(config, out_name, task_name=f"LoRA {op_name}")
        return f"LoRA {op_name} task '{task_id}' added to queue. Output will be {out_name}"
    except Exception as e:
        return f"Error queuing LoRA {op_name}: {e}"
