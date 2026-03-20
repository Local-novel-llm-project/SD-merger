import gradio as gr


SAMPLER_CHOICES = [
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
]

SCHEDULER_CHOICES = [
    "normal",
    "karras",
    "exponential",
    "simple",
    "ddim_uniform",
]

DEFAULT_GENERATION_SETTINGS = {
    "prompt": "A beautiful landscape, high quality, highly detailed, 8k resolution, masterpiece",
    "negative_prompt": "blurry, bad quality, low res, worst quality",
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg": 7.0,
    "sampler_name": "euler",
    "scheduler": "normal",
    "seed": -1,
}


def resolve_generation_settings(defaults=None):
    settings = dict(DEFAULT_GENERATION_SETTINGS)
    if defaults:
        settings.update(defaults)
    return settings


def render_generation_settings_inputs(
    defaults=None,
    *,
    seed_label="Seed (-1 or 0 for random)",
):
    settings = resolve_generation_settings(defaults)

    prompt = gr.Textbox(
        label="Prompt",
        lines=3,
        value=settings["prompt"],
    )
    negative_prompt = gr.Textbox(
        label="Negative Prompt",
        lines=2,
        value=settings["negative_prompt"],
    )

    with gr.Row():
        width = gr.Slider(
            label="Width",
            minimum=256,
            maximum=2048,
            step=64,
            value=settings["width"],
        )
        height = gr.Slider(
            label="Height",
            minimum=256,
            maximum=2048,
            step=64,
            value=settings["height"],
        )

    with gr.Row():
        steps = gr.Slider(
            label="Steps",
            minimum=1,
            maximum=150,
            step=1,
            value=settings["steps"],
        )
        cfg = gr.Slider(
            label="CFG Scale",
            minimum=1.0,
            maximum=30.0,
            step=0.5,
            value=settings["cfg"],
        )

    with gr.Row():
        sampler = gr.Dropdown(
            label="Sampler",
            choices=SAMPLER_CHOICES,
            value=settings["sampler_name"],
        )
        scheduler = gr.Dropdown(
            label="Scheduler",
            choices=SCHEDULER_CHOICES,
            value=settings["scheduler"],
        )

    seed = gr.Number(label=seed_label, value=settings["seed"], precision=0)

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "seed": seed,
    }
