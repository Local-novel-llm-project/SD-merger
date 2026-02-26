import os
import gradio as gr
from module.generation import generate_image
from ui.utils import get_model_list, get_model_path


def render_generation_tab():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Test Generation (Powered by ComfyUI Backend)")

            model_file = gr.Dropdown(
                label="Model to Test", choices=get_model_list()
            )

            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="A beautiful landscape, high quality, highly detailed, 8k resolution, masterpiece",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                value="blurry, bad quality, low res, worst quality",
            )

            with gr.Row():
                width = gr.Slider(
                    label="Width", minimum=256, maximum=2048, step=64, value=512
                )
                height = gr.Slider(
                    label="Height", minimum=256, maximum=2048, step=64, value=512
                )

            with gr.Row():
                steps = gr.Slider(
                    label="Steps", minimum=1, maximum=150, step=1, value=20
                )
                cfg = gr.Slider(
                    label="CFG Scale", minimum=1.0, maximum=30.0, step=0.5, value=7.0
                )

            with gr.Row():
                sampler = gr.Dropdown(
                    label="Sampler",
                    choices=[
                        "euler",
                        "euler_ancestral",
                        "heun",
                        "dpm_2",
                        "dpm_2_ancestral",
                        "lms",
                        "dpm_fast",
                        "dpm_adaptive",
                        "dpmpp_2s_ancestral",
                        "dpmpp_sde",
                        "dpmpp_2m",
                        "ddim",
                        "uni_pc",
                        "uni_pc_bh2",
                    ],
                    value="euler",
                )
                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=[
                        "normal",
                        "karras",
                        "exponential",
                        "simple",
                        "ddim_uniform",
                    ],
                    value="normal",
                )

            seed = gr.Number(label="Seed (-1 or 0 for random)", value=-1, precision=0)

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image")
            output_log = gr.Textbox(label="Log", lines=3)

    def run_generation(m_file, p, n_p, w, h, s, c, samp, sched, se):
        if not m_file:
            return None, "Please select a model file."

        try:
            import random

            actual_seed = (
                int(se) if int(se) > 0 else random.randint(1, 1125899906842624)
            )

            img = generate_image(
                model_path=get_model_path(m_file),
                prompt=p,
                negative_prompt=n_p,
                width=int(w),
                height=int(h),
                steps=int(s),
                cfg=float(c),
                sampler_name=samp,
                scheduler=sched,
                seed=actual_seed,
            )

            if img:
                return img, f"Generated successfully with seed {actual_seed}"
            else:
                return None, "Generation failed. Check console logs for more details."

        except Exception as e:
            return None, f"Error: {e}"

    generate_btn.click(
        run_generation,
        inputs=[
            model_file,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            cfg,
            sampler,
            scheduler,
            seed,
        ],
        outputs=[output_image, output_log],
    )
