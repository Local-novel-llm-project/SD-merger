import os
import gradio as gr
from module.error_messages import build_user_error_message
from module.pipeline.poison import (
    list_poison_preview_entries,
    resolve_poison_output_name,
)
from ui.components.generation_controls import (
    DEFAULT_GENERATION_SETTINGS,
    render_generation_settings_inputs,
)
from ui.utils import enqueue_merge_task, get_model_list, get_model_path


POISON_PREVIEW_POLL_SECONDS = 4


def resolve_poison_output_dir(output_dir: str) -> str:
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    normalized_output_dir = str(output_dir or "").strip() or "models/output/poison_merge"
    if os.path.isabs(normalized_output_dir):
        return normalized_output_dir
    return os.path.abspath(os.path.join(project_root, normalized_output_dir))


def build_poison_merge_task_config(
    base_model_name,
    lora_model_names,
    iterations,
    decay_type,
    initial_alpha,
    alpha_overrides,
    output_dir,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    sampler,
    scheduler,
    seed,
):
    resolved_loras = [get_model_path(model_name) for model_name in (lora_model_names or [])]
    poison_config = {
        "base_model": get_model_path(base_model_name),
        "lora_model": resolved_loras[0] if resolved_loras else None,
        "lora_models": resolved_loras,
        "iterations": int(iterations),
        "decay_type": decay_type,
        "initial_alpha": float(initial_alpha),
        "alpha_overrides": str(alpha_overrides or "").strip(),
        "output_dir": resolve_poison_output_dir(output_dir),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "cfg": float(cfg),
        "sampler_name": sampler,
        "scheduler": scheduler,
        "seed": int(seed),
    }
    output_name = resolve_poison_output_name(poison_config)
    return {
        "output_name": output_name,
        "poison_merge": poison_config,
    }, output_name


def refresh_poison_preview_entries(output_dir):
    abs_out_dir = resolve_poison_output_dir(output_dir)
    preview_entries = list_poison_preview_entries(abs_out_dir)
    if preview_entries:
        return (
            preview_entries,
            f"Loaded {len(preview_entries)} preview tab(s) from {abs_out_dir}",
        )
    if not os.path.isdir(abs_out_dir):
        return [], f"Preview directory not created yet: {abs_out_dir}"
    return [], f"No preview images found in {abs_out_dir}"


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

            preview_defaults = dict(DEFAULT_GENERATION_SETTINGS)
            preview_defaults.update(
                {
                    "prompt": "A beautiful portrait of a character, high quality, masterpiece",
                    "negative_prompt": "blurry, low quality, bad anatomy",
                }
            )
            with gr.Accordion("Preview Generation Settings", open=False):
                preview_inputs = render_generation_settings_inputs(preview_defaults)
                prompt = preview_inputs["prompt"]
                negative_prompt = preview_inputs["negative_prompt"]
                width = preview_inputs["width"]
                height = preview_inputs["height"]
                steps = preview_inputs["steps"]
                cfg = preview_inputs["cfg"]
                sampler = preview_inputs["sampler"]
                scheduler = preview_inputs["scheduler"]
                seed = preview_inputs["seed"]

            run_btn = gr.Button("Run Poison Merge", variant="primary")
            run_log = gr.Textbox(label="Log", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Instructions")
            gr.Markdown(
                """
            1. Select a Base Checkpoint and one or more LoRAs.
            2. Choose how many repetitions you want to perform.
            3. Each iteration applies the selected LoRAs together with the same decaying alpha, using the previous iteration output as the next input.
            4. Images are generated locally at each step to help you evaluate the result.
            5. Since this runs in the Tasks Queue, check the *Tasks Queue* tab for live progress.
            """
            )
            with gr.Row():
                refresh_preview_btn = gr.Button("Refresh Previews", variant="secondary")
            preview_status = gr.Textbox(
                label="Preview Status",
                value="Preview images will appear here after the queued task starts generating them.",
                interactive=False,
            )
            preview_entries_state = gr.State([])
            preview_timer = gr.Timer(value=POISON_PREVIEW_POLL_SECONDS)

            @gr.render(inputs=[preview_entries_state], queue=False, show_progress="hidden")
            def render_preview_tabs(preview_entries):
                gr.Markdown("### Preview Images")
                if not preview_entries:
                    gr.Markdown(
                        "No preview images yet. Run Poison Merge and this panel will populate automatically."
                    )
                    return

                with gr.Tabs():
                    for entry in preview_entries:
                        with gr.TabItem(entry["label"], id=entry["label"]):
                            gr.Image(
                                value=entry["path"],
                                label=entry["label"],
                                interactive=False,
                                show_download_button=True,
                            )

    def run_poison(
        bm,
        lm,
        iters,
        decay,
        alpha,
        overrides,
        out_dir,
        p,
        np,
        w,
        h,
        st,
        c,
        samp,
        sched,
        s,
    ):
        if not bm or not lm:
            return "Base Model and at least one LoRA are required."

        config, output_name = build_poison_merge_task_config(
            bm,
            lm,
            iters,
            decay,
            alpha,
            overrides,
            out_dir,
            p,
            np,
            w,
            h,
            st,
            c,
            samp,
            sched,
            s,
        )

        try:
            task_id = enqueue_merge_task(
                config,
                output_name=output_name,
                task_name=f"Poison Merge ({iters} iters, {len(lm or [])} LoRAs)",
            )
            return f"Poison Merge task '{task_id}' queued successfully. Check 'Tasks Queue' tab for progress."
        except Exception as e:
            return build_user_error_message(e, action="Poison Merge タスクの追加")

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
            width,
            height,
            steps,
            cfg,
            sampler,
            scheduler,
            seed,
        ],
        outputs=[run_log],
    ).then(
        refresh_poison_preview_entries,
        inputs=[output_dir],
        outputs=[preview_entries_state, preview_status],
        queue=False,
    )

    output_dir.change(
        refresh_poison_preview_entries,
        inputs=[output_dir],
        outputs=[preview_entries_state, preview_status],
        queue=False,
    )
    refresh_preview_btn.click(
        refresh_poison_preview_entries,
        inputs=[output_dir],
        outputs=[preview_entries_state, preview_status],
        queue=False,
    )
    preview_timer.tick(
        refresh_poison_preview_entries,
        inputs=[output_dir],
        outputs=[preview_entries_state, preview_status],
        queue=False,
        show_progress="hidden",
    )
