import os
import gradio as gr
import logging

from ui.components.mbw_each import render_mbw_each_tab
from ui.components.multi_merge import render_multi_merge_tab
from ui.components.generation import render_generation_tab
from ui.components.analysis import render_analysis_tab
from ui.components.history import render_history_tab
from module.history import save_history
from ui.components.xyz_plot import render_xyz_plot_tab
from ui.components.elemental_merge import render_elemental_merge_tab


def create_ui():
    """Gradio UI のメインアプリケーションを構築する"""

    with gr.Blocks(title="SD-merger UI") as app:
        gr.Markdown("# SD-merger")
        gr.Markdown("高機能・メモリ効率の高い sd-mecha ベースのモデルマージツール")

        with gr.Tabs():
            # タブ 1: 基本的なマージ (Supermerger風)
            with gr.TabItem("Merge Models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_a = gr.File(
                            label="Model A (Left)", file_types=[".safetensors", ".ckpt"]
                        )
                        model_b = gr.File(
                            label="Model B (Right)",
                            file_types=[".safetensors", ".ckpt"],
                        )
                        model_c = gr.File(
                            label="Model C (Base/Target, optional)",
                            file_types=[".safetensors", ".ckpt"],
                        )

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
                        mbw_str = gr.Textbox(
                            label="Merge Block Weight (MBW)",
                            placeholder="e.g. 1,0.5,0.5,0...",
                        )
                        bake_in_vae = gr.File(
                            label="Bake in VAE", file_types=[".safetensors", ".pt"]
                        )
                        output_name = gr.Textbox(
                            label="Output Filename", value="merged_model.safetensors"
                        )

                merge_btn = gr.Button("Merge Models", variant="primary")
                merge_output = gr.Textbox(label="Output Log")

                def run_merge(a, b, c, strat, t_strat, vel, mbw, vae, out):
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
                    config = {
                        "target_model": c.name if c else a.name,
                        "models": [
                            {
                                "left": a.name,
                                "right": b.name,
                                "strategy": strat,
                                "target_strategy": t_strat,
                                "velocity": float(vel),
                                "key_patterns": ["."],
                            }
                        ],
                    }
                    if mbw:
                        config["models"][0]["mbw"] = mbw
                    if vae:
                        config["bake_in_vae"] = vae.name

                    try:
                        with tempfile.NamedTemporaryFile(
                            "w", delete=False, suffix=".yaml"
                        ) as f:
                            yaml.dump(config, f)
                            tmp_cfg = f.name

                        # 実行
                        out_dir = os.path.abspath("./merged")
                        merger_main(tmp_cfg, out_dir)
                        save_history(
                            {"config": config, "output_name": out, "status": "Success"}
                        )
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
                gr.Markdown("### Extract & Merge LoRAs")
                gr.Markdown("*UI implementation pending...*")

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

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_ui()
    app.launch()
