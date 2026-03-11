import os
import gradio as gr
from ui.utils import get_model_list, get_model_path


def render_poison_merge_tab():
    gr.Markdown("### Poison Merge")
    gr.Markdown(
        "Iteratively apply a LoRA to a base model with a decaying alpha, generating images at each step to find the 'poison' threshold."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_list = get_model_list()
            base_model = gr.Dropdown(label="Base Model", choices=model_list)
            lora_models = gr.Dropdown(
                label="LoRA Models",
                choices=model_list,
                multiselect=True,
            )

            iterations = gr.Slider(
                label="Iterations", minimum=1, maximum=20, step=1, value=3
            )
            decay_type = gr.Dropdown(
                label="Decay Curve",
                choices=["linear", "exponential", "cosine"],
                value="linear",
            )
            initial_alpha = gr.Slider(
                label="Initial Alpha (Weight)",
                minimum=0.01,
                maximum=2.0,
                step=0.01,
                value=1.0,
            )
            alpha_override = gr.Textbox(
                label="Alpha Overrides (comma separated)",
                placeholder="e.g. 1.0, 0.8, 0.5 (Overrides the curve above if provided)",
                value="",
            )

            output_dir = gr.Textbox(label="Output Directory", value="models/output/poison_merge")

            with gr.Accordion("Generation Settings", open=False):
                prompt = gr.Textbox(
                    label="Prompt", value="A beautiful portrait of a character, high quality, masterpiece"
                )
                negative_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality, bad anatomy")
                seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

            run_btn = gr.Button("Run Poison Merge", variant="primary")
            run_log = gr.Textbox(label="Log", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Instructions")
            gr.Markdown(
                """
            1. Select a Base Checkpoint and one or more LoRAs.
            2. Choose how many repetitions you want to perform.
            3. Each iteration applies the selected LoRAs in order with the same decaying alpha, using the output of the previous stage as the next input.
            4. Images are generated locally at each step to help you evaluate the result.
            5. Since this runs in the Tasks Queue, check the *Tasks Queue* tab for live progress.
            """
            )

    def run_poison(bm, lm, iters, decay, alpha, overrides, out_dir, p, np, s):
        if not bm or not lm:
            return "Base Model and at least one LoRA are required."

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        abs_out_dir = (
            os.path.abspath(os.path.join(project_root, out_dir))
            if not os.path.isabs(out_dir)
            else out_dir
        )
        resolved_loras = [get_model_path(model_name) for model_name in lm]

        config = {
            "poison_merge": {
                "base_model": get_model_path(bm),
                "lora_model": resolved_loras[0],
                "lora_models": resolved_loras,
                "iterations": int(iters),
                "decay_type": decay,
                "initial_alpha": float(alpha),
                "alpha_overrides": overrides.strip(),
                "output_dir": abs_out_dir,
                "prompt": p,
                "negative_prompt": np,
                "seed": int(s),
            }
        }

        from module.queue_manager import queue_manager

        try:
            task_id = queue_manager.add_task(
                config,
                output_name="poison_merge",
                task_name=f"Poison Merge ({iters} iters, {len(resolved_loras)} LoRAs)",
            )
            return f"Poison Merge task '{task_id}' queued successfully. Check 'Tasks Queue' tab for progress."
        except Exception as e:
            return f"Error queuing Poison Merge: {e}"

    run_btn.click(
        run_poison,
        inputs=[
            base_model,
            lora_models,
            iterations,
            decay_type,
            initial_alpha,
            alpha_override,
            output_dir,
            prompt,
            negative_prompt,
            seed,
        ],
        outputs=[run_log],
    )
