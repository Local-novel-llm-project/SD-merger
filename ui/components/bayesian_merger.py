import gradio as gr
import logging
import threading

# SD-merger utils
from ui.components.common import create_model_dropdown_pair
from ui.utils import get_model_path

# Bayesian Optimizer modules
from module.bayesian_optimizer.bayes_optimiser import BayesOptimiser
from module.bayesian_optimizer.tpe_optimiser import TPEOptimiser
from module.bayesian_optimizer.atpe_optimiser import ATPEOptimiser
from module.bayesian_optimizer.artist import convergence_plot

BACKGROUND_RUNNER = None


def run_optimization_thread(cfg):
    """バックグラウンドスレッドで最適化を実行する"""
    try:
        opt_type = cfg.get("optimiser", "bayes")
        if opt_type == "bayes":
            optimizer = BayesOptimiser(cfg)
        elif opt_type == "tpe":
            optimizer = TPEOptimiser(cfg)
        elif opt_type == "atpe":
            optimizer = ATPEOptimiser(cfg)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        logging.info(f"Starting {opt_type} optimization...")
        optimizer.optimise()
        optimizer.postprocess()

        # Plot
        try:
            if hasattr(optimizer, "optimizer") and hasattr(optimizer.optimizer, "res"):
                scores = [r["target"] for r in optimizer.optimizer.res]
                plot_path = optimizer.output_dir / f"{optimizer.log_name}_plot.png"
                convergence_plot(scores, figname=plot_path, minimise=False)
                logging.info("Generated convergence plot.")
            elif hasattr(optimizer, "trials"):
                if opt_type == "atpe":
                    scores = [10 - loss for loss in optimizer.trials.losses()]
                else:
                    scores = [-loss for loss in optimizer.trials.losses()]
                plot_path = optimizer.output_dir / f"{optimizer.log_name}_plot.png"
                convergence_plot(scores, figname=plot_path, minimise=False)
                logging.info("Generated convergence plot.")
        except Exception as e:
            logging.error(f"Plotting error: {e}")

    except Exception as e:
        logging.error(f"Optimization failed: {e}")


def create_bayesian_merger_ui():
    """Bayesian Merger タブのUIを構築する"""
    with gr.Blocks() as bayesian_merger_block:
        gr.Markdown("## Bayesian Merger")
        gr.Markdown(
            "自動的にモデルマージ比率（MBW）を探索し、画像生成結果を Aesthetic Score で評価して最適な比率を見つけます。"
        )

        with gr.Row():
            with gr.Column(scale=2):
                model_a, model_b = create_model_dropdown_pair()

                with gr.Accordion("生成・スコアリング設定", open=True):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="1girl, masterpiece, best quality, highly detailed",
                        lines=2,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="worst quality, low quality",
                        lines=2,
                    )

                    with gr.Row():
                        steps = gr.Slider(
                            minimum=1, maximum=150, value=20, step=1, label="Steps"
                        )
                        cfg_scale = gr.Slider(
                            minimum=1.0,
                            maximum=30.0,
                            value=7.0,
                            step=0.1,
                            label="CFG Scale",
                        )
                        seed = gr.Number(
                            label="Seed (-1 for random)", value=-1, precision=0
                        )

                    with gr.Row():
                        width = gr.Slider(
                            minimum=256, maximum=2048, step=8, value=512, label="Width"
                        )
                        height = gr.Slider(
                            minimum=256, maximum=2048, step=8, value=512, label="Height"
                        )

            with gr.Column(scale=1):
                with gr.Accordion("最適化設定", open=True):
                    optimiser_choice = gr.Dropdown(
                        choices=["bayes", "tpe", "atpe"],
                        value="bayes",
                        label="Optimizer Type",
                    )
                    init_points = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=5,
                        step=1,
                        label="Initial Points (Warmup)",
                    )
                    n_iters = gr.Slider(
                        minimum=1, maximum=200, value=20, step=1, label="Iterations"
                    )

                    save_best = gr.Checkbox(
                        label="Save Best Model after Optimization", value=False
                    )
                    output_name = gr.Textbox(
                        label="Best Model Name", value="best_bayesian_merge"
                    )

                btn_run = gr.Button("Run Optimization", variant="primary")
                status = gr.Textbox(label="Status", value="Ready", interactive=False)

        # パラメータ設定 (高度)
        with gr.Accordion(
            "Advanced Parameter Settings (Freeze & Grouping)", open=False
        ):
            gr.Markdown(
                "特定のブロックを固定値にしたり、グループ化して探索次元を減らすことができます。現在はYAMLまたは直書きのサポートを想定しています。"
            )
            custom_params = gr.JSON(
                label="Custom Ranges / Freeze / Group (JSON)",
                value={"frozen_params": {}, "groups": []},
            )

        def start_optimization(
            ma, mb, p, np, s, c, sd, w, h, opt, init, iters, sb, out, custom
        ):
            global BACKGROUND_RUNNER

            if not ma or not mb:
                return "Error: Please select both Model A and Model B."

            cfg = {
                "model_a": get_model_path(ma),
                "model_b": get_model_path(mb),
                "prompts": [p],
                "negative_prompt": np,
                "steps": s,
                "cfg_scale": c,
                "seed": sd,
                "width": w,
                "height": h,
                "optimiser": opt,
                "init_points": init,
                "n_iters": iters,
                "save_best": sb,
                "output_name": out,
            }
            if custom:
                if "frozen_params" in custom:
                    cfg["frozen_params"] = custom["frozen_params"]
                if "groups" in custom:
                    cfg["groups"] = custom["groups"]
                if "custom_ranges" in custom:
                    cfg["custom_ranges"] = custom["custom_ranges"]

            if BACKGROUND_RUNNER and BACKGROUND_RUNNER.is_alive():
                return "Error: Optimization is already running in the background."

            BACKGROUND_RUNNER = threading.Thread(
                target=run_optimization_thread, args=(cfg,), daemon=True
            )
            BACKGROUND_RUNNER.start()

            return "Optimization started in background. Check terminal for progress and `output/bayesian/` for results."

        btn_run.click(
            start_optimization,
            inputs=[
                model_a,
                model_b,
                prompt,
                negative_prompt,
                steps,
                cfg_scale,
                seed,
                width,
                height,
                optimiser_choice,
                init_points,
                n_iters,
                save_best,
                output_name,
                custom_params,
            ],
            outputs=[status],
        )

    return bayesian_merger_block
