import os
import gradio as gr
import yaml
from PIL import Image, ImageDraw
from ui.utils import get_model_list, get_model_path


def render_xyz_plot_tab():
    gr.Markdown("### XY Plot Generation")
    gr.Markdown("Test merge parameters by generating a grid of images.")

    with gr.Row():
        with gr.Column(scale=1):
            model_list = get_model_list()
            model_a = gr.Dropdown(label="Model A (Left)", choices=model_list)
            model_b = gr.Dropdown(label="Model B (Right)", choices=model_list)

            x_type = gr.Dropdown(
                label="X Type",
                choices=["Velocity", "Strategy", "CFG Scale", "Steps", "Model Order"],
                value="Velocity",
            )
            x_values = gr.Textbox(
                label="X Values (comma separated)", value="0.25, 0.5, 0.75"
            )

            y_type = gr.Dropdown(
                label="Y Type",
                choices=[
                    "Velocity",
                    "Strategy",
                    "CFG Scale",
                    "Steps",
                    "Model Order",
                    "None",
                ],
                value="Strategy",
            )
            y_values = gr.Textbox(
                label="Y Values (comma separated)", value="mix, addition"
            )

            z_type = gr.Dropdown(
                label="Z Type",
                choices=[
                    "Velocity",
                    "Strategy",
                    "CFG Scale",
                    "Steps",
                    "Model Order",
                    "None",
                ],
                value="None",
            )
            z_values = gr.Textbox(label="Z Values (comma separated)", value="")

            with gr.Accordion("Generation Settings", open=False):
                prompt = gr.Textbox(label="Prompt", value="A beautiful landscape")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", value="blurry, low quality"
                )
                width = gr.Slider(
                    label="Width", minimum=256, maximum=1024, step=64, value=512
                )
                height = gr.Slider(
                    label="Height", minimum=256, maximum=1024, step=64, value=512
                )
                fixed_seed = gr.Number(
                    label="Seed (-1 or 0 for random)", value=-1, precision=0
                )

            generate_btn = gr.Button("Generate XY Grid", variant="primary")
            output_log = gr.Textbox(label="Log", interactive=False)

        with gr.Column(scale=2):
            output_grid = gr.Image(label="XY(Z) Grid", type="filepath")

    def run_xy(ma, mb, xt, xv, yt, yv, zt, zv, p, np, w, h, seed_in):
        if not ma or not mb:
            return None, "Model A and Model B required."

        import sys
        import random

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        )
        from main import main as merger_main
        from module.generation import generate_first_image
        from module.extension_manager import load_extensions

        load_extensions()

        x_vals = [x.strip() for x in xv.split(",") if x.strip()]
        y_vals = (
            [y.strip() for y in yv.split(",") if y.strip()]
            if yt != "None"
            else ["None"]
        )
        z_vals = (
            [z.strip() for z in zv.split(",") if z.strip()]
            if zt != "None"
            else ["None"]
        )

        actual_seed = (
            int(seed_in) if int(seed_in) > 0 else random.randint(1, 1125899906842624)
        )
        w = int(w)
        h = int(h)

        def parse_val(vtype, val):
            if val == "None":
                return None
            if vtype in ["Velocity", "CFG Scale"]:
                return float(val)
            if vtype in ["Steps"]:
                return int(val)
            return val

        log = f"Using Seed: {actual_seed}\n"

        tmp_dir = os.path.abspath("./merged/xyz_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        frames = []

        for z_idx, z_raw in enumerate(z_vals):
            z_val = parse_val(zt, z_raw)
            images = []

            for y_idx, y_raw in enumerate(y_vals):
                y_val = parse_val(yt, y_raw)
                for x_idx, x_raw in enumerate(x_vals):
                    x_val = parse_val(xt, x_raw)

                    velocity, strategy, cfg, steps = 0.5, "mix", 7.0, 20

                    if xt == "Velocity":
                        velocity = x_val
                    if yt == "Velocity":
                        velocity = y_val
                    if zt == "Velocity":
                        velocity = z_val

                    if xt == "Strategy":
                        strategy = x_val
                    if yt == "Strategy":
                        strategy = y_val
                    if zt == "Strategy":
                        strategy = z_val

                    if xt == "CFG Scale":
                        cfg = x_val
                    if yt == "CFG Scale":
                        cfg = y_val
                    if zt == "CFG Scale":
                        cfg = z_val

                    if xt == "Steps":
                        steps = x_val
                    if yt == "Steps":
                        steps = y_val
                    if zt == "Steps":
                        steps = z_val

                    left_model = get_model_path(ma)
                    right_model = get_model_path(mb)

                    model_order = "A->B"
                    if xt == "Model Order":
                        model_order = str(x_val).strip()
                    if yt == "Model Order":
                        model_order = str(y_val).strip()
                    if zt == "Model Order":
                        model_order = str(z_val).strip()

                    if model_order == "B->A":
                        left_model, right_model = right_model, left_model

                    merge_params = [xt, yt, zt]
                    if any(
                        p in ["Velocity", "Strategy", "Model Order"]
                        for p in merge_params
                    ):
                        config = {
                            "target_model": left_model,
                            "models": [
                                {
                                    "left": left_model,
                                    "right": right_model,
                                    "strategy": strategy,
                                    "velocity": velocity,
                                    "key_patterns": ["."],
                                }
                            ],
                        }
                        cfg_file = os.path.join(
                            tmp_dir, f"cfg_{z_idx}_{y_idx}_{x_idx}.yaml"
                        )
                        with open(cfg_file, "w") as f:
                            yaml.dump(config, f)

                        import glob

                        for stale_file in glob.glob(
                            os.path.join(tmp_dir, "*.safetensors")
                        ):
                            os.remove(stale_file)

                        log += (
                            f"Merging for {xt}={x_val}, {yt}={y_val}, {zt}={z_val}...\n"
                        )
                        merger_main(cfg_file, tmp_dir)

                        files = glob.glob(os.path.join(tmp_dir, "*.safetensors"))
                        if not files:
                            return (
                                None,
                                log
                                + f"\nFailed to create merged model for {x_val}, {y_val}, {z_val}",
                            )

                        out_model = max(files, key=os.path.getctime)
                    else:
                        out_model = get_model_path(ma)

                    log += f"Generating image for {xt}={x_val}, {yt}={y_val}, {zt}={z_val}...\n"
                    img = generate_first_image(
                        model_path=out_model,
                        prompt=p,
                        negative_prompt=np,
                        width=w,
                        height=h,
                        steps=steps,
                        cfg=cfg,
                        sampler_name="euler",
                        scheduler="normal",
                        seed=actual_seed,
                    )

                    if img is None:
                        return (
                            None,
                            log + f"\nFailed to generate for {x_val}, {y_val}, {z_val}",
                        )

                    images.append(img)

            grid_w = len(x_vals) * w
            grid_h = len(y_vals) * h
            grid_img = Image.new("RGB", (grid_w, grid_h))

            draw = ImageDraw.Draw(grid_img)

            for y_idy in range(len(y_vals)):
                for x_idx in range(len(x_vals)):
                    idx = y_idy * len(x_vals) + x_idx
                    px = x_idx * w
                    py = y_idy * h
                    grid_img.paste(images[idx], (px, py))

                    # Draw labels
                    text_parts = [f"{xt}={x_vals[x_idx]}"]
                    if yt != "None":
                        text_parts.append(f"{yt}={y_vals[y_idy]}")
                    if zt != "None":
                        text_parts.append(f"{zt}={z_val}")

                    text = ", ".join(text_parts)
                    draw.text((px + 10, py + 10), text, fill="white")

            frames.append(grid_img)

        # Output logic
        if not frames:
            return None, log + "\nNo images generated."

        out_path = os.path.abspath(
            os.path.join("./merged", f"xyz_grid_{actual_seed}.gif")
        )
        if len(frames) == 1:
            out_path = out_path.replace(".gif", ".png")
            frames[0].save(out_path)
        else:
            frames[0].save(
                out_path, save_all=True, append_images=frames[1:], duration=1500, loop=0
            )

        log += "Grid completed successfully!"
        return out_path, log

    generate_btn.click(
        run_xy,
        inputs=[
            model_a,
            model_b,
            x_type,
            x_values,
            y_type,
            y_values,
            z_type,
            z_values,
            prompt,
            negative_prompt,
            width,
            height,
            fixed_seed,
        ],
        outputs=[output_grid, output_log],
    )
