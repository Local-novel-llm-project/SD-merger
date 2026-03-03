import os
import gradio as gr
import yaml

from ui.utils import get_model_list, get_model_path
from module.generation import generate_image
from module.metrics import calculate_clip_score, generate_radar_chart


def render_ab_test_tab():
    gr.Markdown("### A/B Test Merge")
    gr.Markdown(
        "Compare Model A, Model B, and their merged outputs (A->B and B->A) under identical generation conditions."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_a = gr.Dropdown(label="Model A", choices=get_model_list(), scale=1)
            model_b = gr.Dropdown(label="Model B", choices=get_model_list(), scale=1)

            strategy = gr.Dropdown(
                label="Merge Strategy",
                choices=["addition", "subtraction", "multiplication", "mix", "cosineA", "cosineB", "smoothAdd"],
                value="mix",
            )
            velocity = gr.Slider(label="Velocity (alpha)", minimum=0.0, maximum=1.0, step=0.01, value=0.5)

            with gr.Accordion("Generation Settings", open=True):
                prompt = gr.Textbox(label="Prompt", value="A highly detailed masterpiece, best quality")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="worst quality, bad, blurry")
                seed = gr.Number(label="Seed", value=12345, precision=0)
                width = gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=512)
                height = gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=512)
                steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=20)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=30.0, step=0.5, value=7.0)

            generate_btn = gr.Button("Generate Comparison", variant="primary")
            metrics_btn = gr.Button("Calculate Metrics (CLIP Score)")
            output_log = gr.Textbox(label="Log", interactive=False)

        with gr.Column(scale=2):
            with gr.Row():
                img_a = gr.Image(label="Model A")
                img_b = gr.Image(label="Model B")
            with gr.Row():
                img_ab = gr.Image(label="Merged (A -> B)")
                img_ba = gr.Image(label="Merged (B -> A)")

            with gr.Row():
                radar_plot = gr.Image(label="Metrics Chart")

            # Hidden state to store generated images for metric calculation
            state_images = gr.State({})

    def run_comparison(ma, mb, strat, vel, p, np, s, w, h, st, c):
        if not ma or not mb:
            return None, None, None, None, {}, "Please select Model A and Model B."

        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from main import main as merger_main

        log = f"Starting comparison with seed: {s}\n"
        images_generated = {}

        def _generate(model_path, title):
            log_str = f"Generating {title}...\n"
            img = generate_image(
                model_path=model_path,
                prompt=p,
                negative_prompt=np,
                width=int(w),
                height=int(h),
                steps=int(st),
                cfg=float(c),
                seed=int(s),
            )
            return img, log_str

        # Generate base models
        img_a_res, l_a = _generate(get_model_path(ma), "Model A")
        img_b_res, l_b = _generate(get_model_path(mb), "Model B")
        log += l_a + l_b
        images_generated["Model A"] = img_a_res
        images_generated["Model B"] = img_b_res

        tmp_dir = os.path.abspath("./merged/ab_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        def _merge_and_generate(left_name, right_name, title):
            config = {
                "target_model": get_model_path(left_name),
                "models": [
                    {
                        "left": get_model_path(left_name),
                        "right": get_model_path(right_name),
                        "strategy": strat,
                        "velocity": float(vel),
                        "key_patterns": ["."],
                    }
                ],
            }
            cfg_file = os.path.join(tmp_dir, f"cfg_{title.replace(' ', '_')}.yaml")
            with open(cfg_file, "w") as f:
                yaml.dump(config, f)

            merger_main(cfg_file, tmp_dir)
            import glob

            files = glob.glob(os.path.join(tmp_dir, "*.safetensors"))
            out_model = max(files, key=os.path.getctime) if files else get_model_path(left_name)

            img_res, l_res = _generate(out_model, title)
            return img_res, l_res

        # A -> B
        img_ab_res, l_ab = _merge_and_generate(ma, mb, "Merged A->B")
        log += l_ab
        images_generated["Merged (A->B)"] = img_ab_res

        # B -> A
        img_ba_res, l_ba = _merge_and_generate(mb, ma, "Merged B->A")
        log += l_ba
        images_generated["Merged (B->A)"] = img_ba_res

        log += "Generation completed!"
        return img_a_res, img_b_res, img_ab_res, img_ba_res, images_generated, log

    def calculate_current_metrics(images_dict, p):
        if not images_dict or "Model A" not in images_dict or images_dict["Model A"] is None:
            return None, "No valid images generated yet. Please Generate Comparison first."

        log = "Calculating CLIP Scores...\n"
        metrics_data = {}
        for title, img in images_dict.items():
            if img:
                score_list = calculate_clip_score([img], p)
                score = score_list[0] if score_list else 0.0
                metrics_data[title] = {"CLIP Score": score}
                log += f"{title} CLIP Score: {score:.4f}\n"

        if metrics_data:
            # Generate radar chart or a simple bar if it's only 1 metric, but radar handles 1 point via internal duplication trick
            radar = generate_radar_chart(metrics_data, title="A/B Comparison Metrics")
            log += "Metrics calculation completed."
            return radar, log
        return None, "Failed to calculate metrics."

    generate_btn.click(
        run_comparison,
        inputs=[model_a, model_b, strategy, velocity, prompt, negative_prompt, seed, width, height, steps, cfg],
        outputs=[img_a, img_b, img_ab, img_ba, state_images, output_log],
    )

    metrics_btn.click(calculate_current_metrics, inputs=[state_images, prompt], outputs=[radar_plot, output_log])
