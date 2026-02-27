import os
import sys
import gradio as gr
import logging

# Ensure the project root is in sys.path so 'ui' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.components.mbw_each import render_mbw_each_tab
from ui.components.multi_merge import render_multi_merge_tab
from ui.components.generation import render_generation_tab
from ui.components.analysis import render_analysis_tab
from ui.components.history import render_history_tab
from module.history import save_history
from ui.components.xyz_plot import render_xyz_plot_tab
from ui.components.elemental_merge import render_elemental_merge_tab
from ui.components.dice_roll import render_dice_roll_tab
from ui.components.presets import render_presets_tab
from ui.components.lora_ops import render_lora_ops_tab
from ui.utils import get_model_list, get_model_path


def create_ui():
    """Gradio UI のメインアプリケーションを構築する"""
    from module.extension_manager import load_extensions

    load_extensions()

    with gr.Blocks(title="SD-merger UI") as app:
        gr.Markdown("# SD-merger")
        gr.Markdown("高機能・メモリ効率の高い sd-mecha ベースのモデルマージツール")

        with gr.Tabs():
            # タブ 1: 基本的なマージ (Supermerger風)
            with gr.TabItem("Merge Models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_list = get_model_list()
                        model_a = gr.Dropdown(label="Model A (Left)", choices=model_list)
                        model_b = gr.Dropdown(label="Model B (Right)", choices=model_list)
                        model_c = gr.Dropdown(label="Model C (Base/Target, optional)", choices=model_list)

                    with gr.Column(scale=1):
                        strategy = gr.Dropdown(
                            label="Merge Strategy (Left/Right)",
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
                            value="mix",
                        )
                        target_strategy = gr.Dropdown(
                            label="Target Strategy (apply to Model C)",
                            choices=[
                                "mix",
                                "addition",
                                "subtraction",
                                "angle",
                                "trainDifference",
                                "extract",
                            ],
                            value="mix",
                        )
                        velocity = gr.Slider(
                            label="Velocity (alpha)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.5,
                        )

                with gr.Row():
                    with gr.Accordion("Advanced Options", open=False):
                        use_advanced_options = gr.Checkbox(label="Enable Advanced Options", value=False)
                        mbw_str = gr.Textbox(
                            label="Merge Block Weight (MBW)",
                            placeholder="e.g. 1,0.5,0.5,0...",
                        )
                        bake_in_vae = gr.Dropdown(label="Bake in VAE", choices=get_model_list())
                        output_name = gr.Textbox(label="Output Filename", value="merged_model.safetensors")

                merge_btn = gr.Button("Merge Models", variant="primary")
                merge_output = gr.Textbox(label="Output Log")

                def run_merge(a, b, c, strat, t_strat, vel, use_adv, mbw, vae, out):
                    if not a or not b:
                        return "Model A and Model B are required."

                    import sys

                    sys.path.insert(
                        0,
                        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                    )
                    from main import main as merger_main
                    import tempfile
                    import yaml

                    # YAML configを一時ファイルに生成
                    target_model_path = get_model_path(c) if c else get_model_path(a)
                    left_model_path = get_model_path(a)
                    right_model_path = get_model_path(b)

                    config = {
                        "target_model": target_model_path,
                        "models": [
                            {
                                "left": left_model_path,
                                "right": right_model_path,
                                "strategy": strat,
                                "target_strategy": t_strat,
                                "velocity": float(vel),
                                "key_patterns": ["."],
                            }
                        ],
                    }
                    if use_adv and mbw:
                        config["models"][0]["mbw"] = mbw
                    if use_adv and vae:
                        config["bake_in_vae"] = get_model_path(vae)
                    if use_adv and out:
                        config["output_name"] = out

                    try:
                        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
                            yaml.dump(config, f)
                            tmp_cfg = f.name

                        # 実行
                        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "output"))
                        merger_main(tmp_cfg, out_dir)
                        save_history({"config": config, "output_name": out, "status": "Success"})
                        return f"Merge completed successfully. Saved to {out_dir}"
                    except Exception as e:
                        save_history(
                            {
                                "config": config,
                                "output_name": out,
                                "status": f"Failed: {e}",
                            }
                        )
                        return f"Error during merge: {e}"

                merge_btn.click(
                    run_merge,
                    inputs=[
                        model_a,
                        model_b,
                        model_c,
                        strategy,
                        target_strategy,
                        velocity,
                        use_advanced_options,
                        mbw_str,
                        bake_in_vae,
                        output_name,
                    ],
                    outputs=[merge_output],
                )

            # タブ 2: MBW Each
            with gr.TabItem("MBW Each"):
                render_mbw_each_tab()

            # タブ 3: LoRA Operations
            with gr.TabItem("LoRA Ops"):
                render_lora_ops_tab()

            # タブ 4: Multi-Merge (Batch)
            with gr.TabItem("Multi-Merge"):
                render_multi_merge_tab()

            # タブ 5: Generate & Test
            with gr.TabItem("Generate & Test"):
                render_generation_tab()

            # タブ 6: Analysis
            with gr.TabItem("Analysis"):
                render_analysis_tab()

            # タブ 7: History
            with gr.TabItem("History"):
                render_history_tab()

            # タブ 8: XYZ Plot
            with gr.TabItem("XYZ Plot"):
                render_xyz_plot_tab()

            # タブ 9: Visual MBW
            with gr.TabItem("Visual MBW"):
                render_elemental_merge_tab()

            # タブ 10: Let the Dice Roll
            with gr.TabItem("Let the Dice Roll"):
                render_dice_roll_tab()

            # タブ 11: Presets
            with gr.TabItem("Presets"):
                render_presets_tab()

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_ui()
    app.launch()
