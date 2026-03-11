import os
import sys
import gradio as gr

# Ensure the project root is in sys.path so 'ui' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.components.mbw_each import render_mbw_each_tab
from ui.components.multi_merge import render_multi_merge_tab
from ui.components.generation import render_generation_tab
from ui.components.analysis import render_analysis_tab
from ui.components.history import render_history_tab
from ui.components.xyz_plot import render_xyz_plot_tab
from ui.components.elemental_merge import render_elemental_merge_tab
from ui.components.dice_roll import render_dice_roll_tab
from ui.components.presets import render_presets_tab
from ui.components.lora_ops import render_lora_ops_tab
from ui.components.poison_merge import render_poison_merge_tab
from ui.components.ab_test import render_ab_test_tab
from ui.utils import get_model_list, get_model_path

from module.error_messages import build_user_error_message
from ui.components.queue_ui import render_queue_tab
from ui.components.bayesian_merger import create_bayesian_merger_ui
from module.queue_manager import queue_manager


def create_ui():
    """Gradio UI のメインアプリケーションを構築する"""
    from module.extension_manager import load_extensions

    load_extensions()

    # 起動時にキューワーカーを開始
    queue_manager.start_worker()

    with gr.Blocks(
        title="SD-merger UI",
    ) as app:
        gr.Markdown("# SD-merger")
        with gr.Row():
            gr.Markdown("高機能・メモリ効率の高い sd-mecha ベースのモデルマージツール")
            dark_mode_btn = gr.Button("🌓 Toggle Dark Mode", size="sm", scale=0)

            # Use javascript to toggle dark mode class on body
            dark_mode_btn.click(
                None,
                None,
                None,
                js="""
                () => {
                    document.body.classList.toggle('dark');
                    const isDark = document.body.classList.contains('dark');
                    localStorage.setItem('theme', isDark ? 'dark' : 'light');
                }
                """,
            )

        with gr.Tabs():
            # タブ 1: 基本的なマージ (Supermerger風)
            with gr.TabItem("Merge Models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            refresh_model_btn = gr.Button("🔄 Refresh Models", size="sm")

                        model_list = get_model_list()
                        model_a = gr.Dropdown(label="Model A (Left)", choices=model_list)
                        model_b = gr.Dropdown(label="Model B (Right)", choices=model_list)
                        model_c = gr.Dropdown(label="Model C (Base/Target, optional)", choices=["選択しない"] + model_list, value="選択しない")

                        def refresh_dropdowns():
                            updated_list = get_model_list()
                            return [
                                gr.update(choices=updated_list),
                                gr.update(choices=updated_list),
                                gr.update(choices=["選択しない"] + updated_list),
                            ]

                        refresh_model_btn.click(refresh_dropdowns, inputs=[], outputs=[model_a, model_b, model_c])

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

                    if c == "選択しない":
                        target_model_path = ""
                    else:
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
                    else:
                        out = f"queue_{int(vel * 100)}_{strat}.safetensors"

                    try:
                        task_id = queue_manager.add_task(config, out, task_name=f"Merge: {strat}")
                        return f"Merge task '{task_id}' added to queue. Output will be {out}"
                    except Exception as e:
                        return build_user_error_message(e, action="マージタスクの追加")

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
            from ui.components.generation import render_auto_generate_settings

            with gr.TabItem("Generate & Test"):
                render_generation_tab()
                gr.Markdown("---")
                render_auto_generate_settings()

            # タブ 6: Analysis
            with gr.TabItem("Analysis"):
                render_analysis_tab()

            # タブ 7: History
            with gr.TabItem("History"):
                history_refresh_btn, history_rerun_btn, history_table = render_history_tab()

            # タブ 8: XYZ Plot
            with gr.TabItem("XYZ Plot"):
                render_xyz_plot_tab()

            # タブ 8.5: A/B Test Merge
            with gr.TabItem("A/B Test Merge"):
                render_ab_test_tab()

            # タブ 9: Visual MBW
            with gr.TabItem("Visual MBW"):
                render_elemental_merge_tab()

            # タブ 10: Let the Dice Roll
            with gr.TabItem("Let the Dice Roll"):
                render_dice_roll_tab()

            # タブ 11: Presets
            with gr.TabItem("Presets"):
                render_presets_tab()

            # タブ 12: Poison Merge
            with gr.TabItem("Poison Merge"):
                render_poison_merge_tab()

            # タブ 13: Queue Manager
            with gr.TabItem("Tasks Queue"):
                render_queue_tab()

            # タブ 14: Bayesian Merger
            with gr.TabItem("Bayesian Merger"):
                create_bayesian_merger_ui()

    return app


if __name__ == "__main__":
    app = create_ui()
    
    # launch arguments for network exposure and UI styles
    head_content = """
<style>
/* Optional custom CSS overrides for better appearance */
.gradio-container { max-width: 1400px !important; }
</style>
<script>
// Keyboard shortcut handling
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter or Cmd+Enter to trigger the primary button on active tab
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const primaryBtn = document.querySelector('.tabitem[style*="block"] button.primary');
        if (primaryBtn) {
            primaryBtn.click();
            e.preventDefault();
        }
    }
});
</script>
"""
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), head=head_content)
