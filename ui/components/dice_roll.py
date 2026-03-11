import gradio as gr
import os
import random

from module.error_messages import build_user_error_message
from module.generation import generate_first_image
from ui.utils import get_model_list, get_model_path, run_merge_from_config


def render_dice_roll_tab():
    gr.Markdown("### Let the Dice Roll (Random Merge Search)")
    gr.Markdown(
        "Generate a random MBW (Merge Block Weight) string and strategy to discover new model combinations automatically."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_list = get_model_list()
            model_a = gr.Dropdown(label="Model A (Left)", choices=model_list)
            model_b = gr.Dropdown(label="Model B (Right)", choices=model_list)

            with gr.Accordion("Randomization Constraints", open=True):
                strat_options = gr.CheckboxGroup(
                    label="Allowed Strategies",
                    choices=[
                        "addition",
                        "subtraction",
                        "multiplication",
                        "mix",
                        "cosineA",
                        "smoothAdd",
                        "tensor",
                    ],
                    value=["mix", "addition", "cosineA"],
                )
                alpha_min = gr.Slider(
                    label="Minimum Velocity (Alpha)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                )
                alpha_max = gr.Slider(
                    label="Maximum Velocity (Alpha)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                )

            with gr.Accordion("Generation Settings", open=False):
                prompt = gr.Textbox(
                    label="Prompt", value="A beautiful landscape, masterpiece"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", value="blurry, lowres"
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

            roll_btn = gr.Button("🎲 Roll the Dice!", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Result")
            output_params = gr.JSON(label="Rolled Parameters")
            output_log = gr.Textbox(label="Log", interactive=False)

    def run_dice(ma, mb, allowed_strats, a_min, a_max, p, np, w, h, seed_in):
        if not ma or not mb:
            return None, {}, "Model A and Model B are required."
        if not allowed_strats:
            return None, {}, "Please select at least one strategy."

        from module.history import save_history

        # Roll strategy
        strategy = random.choice(allowed_strats)

        # Roll velocity
        velocity = round(random.uniform(a_min, a_max), 3)

        # Roll MBW
        mbw = [round(random.uniform(a_min, a_max), 3) for _ in range(26)]
        mbw_str = ",".join(map(str, mbw))

        actual_seed = (
            int(seed_in) if int(seed_in) > 0 else random.randint(1, 1125899906842624)
        )

        rolled_params = {
            "strategy": strategy,
            "velocity": velocity,
            "mbw": mbw_str,
            "seed": actual_seed,
        }

        # Config
        tmp_dir = os.path.abspath("./merged/dice_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        config = {
            "target_model": get_model_path(ma),
            "models": [
                {
                    "left": get_model_path(ma),
                    "right": get_model_path(mb),
                    "strategy": strategy,
                    "velocity": velocity,
                    "mbw": mbw_str,
                    "key_patterns": ["."],
                }
            ],
        }

        out_model = os.path.join(tmp_dir, "dice_result.safetensors")
        log = f"Rolled: Strategy={strategy}, Velocity={velocity}, Seed={actual_seed}\nMerging...\n"

        try:
            out_model = run_merge_from_config(config, tmp_dir) or out_model

            save_history(
                {
                    "config": config,
                    "output_name": out_model,
                    "status": "Dice Roll Success",
                }
            )

            log += "Generating image...\n"
            img = generate_first_image(
                model_path=out_model,
                prompt=p,
                negative_prompt=np,
                width=int(w),
                height=int(h),
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                seed=actual_seed,
            )

            if img:
                return img, rolled_params, log + "Done!"
            else:
                return None, rolled_params, log + "Image generation failed."

        except Exception as e:
            return None, rolled_params, build_user_error_message(e, action="Dice Roll 実行")

    roll_btn.click(
        run_dice,
        inputs=[
            model_a,
            model_b,
            strat_options,
            alpha_min,
            alpha_max,
            prompt,
            negative_prompt,
            width,
            height,
            fixed_seed,
        ],
        outputs=[output_image, output_params, output_log],
    )
