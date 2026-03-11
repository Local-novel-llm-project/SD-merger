import gradio as gr
from module.error_messages import build_user_error_message, build_user_message
from module.generation import generate_image, clear_model_cache, get_cache_info
from ui.utils import get_model_list, get_model_path


def render_generation_tab():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Test Generation (Powered by Diffusers)")

            with gr.Row():
                model_file = gr.Dropdown(
                    label="Model to Test",
                    choices=get_model_list(),
                    scale=8,
                )
                reload_btn = gr.Button("🔄", scale=1, min_width=60)

            # --- Model Cache Management ---
            with gr.Row():
                cache_info_text = gr.Textbox(label="Model Cache Status", value="Not loaded", interactive=False, scale=4)
                clear_cache_btn = gr.Button("Clear Cache", variant="secondary", scale=1)

            def refresh_models():
                return gr.update(choices=get_model_list())

            def ui_clear_cache():
                count = clear_model_cache()
                return f"Cleared {count} models from cache."

            def update_cache_info():
                info = get_cache_info()
                return f"{info['count']} models loaded ({info['total_gb']:.2f} / {info['max_gb']:.2f} GB)"

            reload_btn.click(fn=refresh_models, inputs=[], outputs=[model_file])

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
                width = gr.Slider(label="Width", minimum=256, maximum=2048, step=64, value=512)
                height = gr.Slider(label="Height", minimum=256, maximum=2048, step=64, value=512)

            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=20)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=30.0, step=0.5, value=7.0)

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
            output_gallery = gr.Gallery(label="Generated Images", columns=2, object_fit="contain", height="auto")
            output_log = gr.Textbox(label="Log", lines=3)

    def run_generation(m_file, p, n_p, w, h, s, c, samp, sched, se):
        if not m_file:
            return None, "Please select a model file."

        try:
            import random

            actual_seed = int(se) if int(se) > 0 else random.randint(1, 1125899906842624)

            images = generate_image(
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

            if images and len(images) > 0:
                return images, f"Generated successfully with seed {actual_seed}"
            else:
                return None, build_user_message(
                    "画像生成",
                    "画像を生成できませんでした。",
                    "モデル形式、プロンプト、サンプラー設定を確認してください。",
                )

        except Exception as e:
            return None, build_user_error_message(e, action="画像生成")

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
        outputs=[output_gallery, output_log],
    ).then(
        update_cache_info,
        inputs=[],
        outputs=[cache_info_text],
    )

    reload_btn.click(fn=refresh_models, inputs=[], outputs=[model_file])
    clear_cache_btn.click(fn=ui_clear_cache, inputs=[], outputs=[cache_info_text])


def render_auto_generate_settings():
    """マージ後の自動生成（拡張機能）の設定UIをレンダリングする"""
    from extensions.auto_generate import auto_generate_config

    with gr.Accordion("Auto Generate (Post Merge Hook)", open=False):
        gr.Markdown("Configure automatic image generation after merging (saves to same directory as model).")
        with gr.Row():
            with gr.Column(scale=1):
                enable_auto_gen = gr.Checkbox(label="Enable Auto Generate", value=auto_generate_config["enabled"])
                ag_prompt = gr.Textbox(label="Prompt", lines=3, value=auto_generate_config["prompt"])
                ag_neg_prompt = gr.Textbox(
                    label="Negative Prompt", lines=2, value=auto_generate_config["negative_prompt"]
                )

                with gr.Row():
                    ag_width = gr.Slider(
                        label="Width", minimum=256, maximum=2048, step=64, value=auto_generate_config["width"]
                    )
                    ag_height = gr.Slider(
                        label="Height", minimum=256, maximum=2048, step=64, value=auto_generate_config["height"]
                    )

                with gr.Row():
                    ag_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=150, step=1, value=auto_generate_config["steps"]
                    )
                    ag_cfg = gr.Slider(
                        label="CFG Scale", minimum=1.0, maximum=30.0, step=0.5, value=auto_generate_config["cfg"]
                    )

                with gr.Row():
                    ag_sampler = gr.Dropdown(
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
                        value=auto_generate_config["sampler_name"],
                    )
                    ag_scheduler = gr.Dropdown(
                        label="Scheduler",
                        choices=["normal", "karras", "exponential", "simple", "ddim_uniform"],
                        value=auto_generate_config["scheduler"],
                    )

                ag_seed = gr.Number(label="Seed (-1 or 0 for random)", value=auto_generate_config["seed"], precision=0)

                # 自動生成設定の更新機構
                def update_auto_gen_settings(en, p, np, w, h, s, c, samp, sched, se):
                    auto_generate_config.update(
                        {
                            "enabled": en,
                            "prompt": p,
                            "negative_prompt": np,
                            "width": int(w),
                            "height": int(h),
                            "steps": int(s),
                            "cfg": float(c),
                            "sampler_name": samp,
                            "scheduler": sched,
                            "seed": int(se),
                        }
                    )
                    return "Auto Generate settings updated."

                update_status = gr.Textbox(label="Status", interactive=False)

                # ユーザーが編集するたびにリアルタイム更新（もしくはボタンで行う）
                # 今回は各コンポーネントの change イベントで更新する設計とする
                inputs = [
                    enable_auto_gen,
                    ag_prompt,
                    ag_neg_prompt,
                    ag_width,
                    ag_height,
                    ag_steps,
                    ag_cfg,
                    ag_sampler,
                    ag_scheduler,
                    ag_seed,
                ]
                for comp in inputs:
                    if hasattr(comp, "change"):
                        comp.change(fn=update_auto_gen_settings, inputs=inputs, outputs=[update_status])

    return [enable_auto_gen]  # 必要であればルートコンポーネント群を返す
