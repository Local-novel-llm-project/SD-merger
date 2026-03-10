import os
import math
import logging
from typing import List, Dict, Any


def calculate_alphas(
    initial_alpha: float, iterations: int, decay_type: str
) -> List[float]:
    """反復マージ用の減衰アルファ値リストを計算する。"""
    alphas = []
    for i in range(iterations):
        if iterations == 1:
            alphas.append(initial_alpha)
            continue

        progress = i / (iterations - 1)
        if decay_type == "linear":
            alpha = initial_alpha * (1.0 - progress)
        elif decay_type == "exponential":
            alpha = initial_alpha * (0.5 ** (progress * 5))  # Example steep curve
        elif decay_type == "cosine":
            alpha = initial_alpha * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            alpha = initial_alpha * (1.0 - progress)  # Default linear
        alphas.append(max(0.001, alpha))  # Avoid strictly zero

    return alphas


def run_poison_merge(config: Dict[str, Any], task_name: str = "Poison Merge") -> None:
    """Poison Merge のメインイテレーションを実行する。

    各イテレーションで以下の処理を行う:
    1. Base モデルに LoRA を適用 (alphaを指定)
    2. 画像を生成して保存
    3. [次ステップへ] 出力されたマージ済みモデルを新たなBaseとして続行
    """
    from main import main as merger_main
    from module.generation import generate_first_image
    from PIL import Image, ImageDraw

    p_config = config["poison_merge"]
    base_model = p_config["base_model"]
    lora_model = p_config["lora_model"]
    iterations = p_config.get("iterations", 3)
    decay_type = p_config.get("decay_type", "linear")
    initial_alpha = p_config.get("initial_alpha", 1.0)
    alpha_overrides_str = p_config.get("alpha_overrides", "")
    output_dir = p_config["output_dir"]
    prompt = p_config.get("prompt", "A beautiful landscape")
    negative_prompt = p_config.get("negative_prompt", "blurry")
    seed = p_config.get("seed", -1)

    alphas = calculate_alphas(initial_alpha, iterations, decay_type)

    # オーバーライドがあればパースして上書き
    if alpha_overrides_str:
        try:
            overrides = [
                float(x.strip()) for x in alpha_overrides_str.split(",") if x.strip()
            ]
            if overrides:
                # 入力された数に合わせてイテレーション回数を調整
                alphas = overrides
                iterations = len(alphas)
                logging.info(f"[{task_name}] Using custom alpha overrides: {alphas}")
        except ValueError:
            logging.warning(
                f"[{task_name}] Invalid alpha overrides format. Falling back to calculated curve."
            )

    logging.info(f"[{task_name}] Target Alphas: {alphas}")

    os.makedirs(output_dir, exist_ok=True)
    images = []

    current_base = base_model

    for i, current_alpha in enumerate(alphas):
        logging.info(
            f"[{task_name}] Step {i + 1}/{iterations}, Alpha: {current_alpha:.3f}"
        )

        # モデル名: <basename>_step<X>.safetensors
        out_name = f"poison_step_{i + 1}_alpha_{current_alpha:.2f}.safetensors"
        out_path = os.path.join(output_dir, out_name)

        # マージ用の設定を動的生成
        import tempfile
        import yaml

        merge_config = {
            "target_model": current_base,
            "models": [
                {
                    "left": current_base,
                    "right": lora_model,
                    "strategy": "replace",
                    "replace_with": "right",
                    "target_strategy": "addition",
                    "velocity": float(current_alpha),
                    "key_patterns": ["."],
                }
            ],
            "output_name": out_name,
        }

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
            yaml.dump(merge_config, f)
            tmp_cfg = f.name

        try:
            # マージ実行
            merger_main(tmp_cfg, output_dir)
            logging.info(f"[{task_name}] Step {i + 1} Merge completed: {out_name}")

            # 画像生成
            import random

            actual_seed = (
                int(seed) if int(seed) > 0 else random.randint(1, 1125899906842624)
            )
            logging.info(
                f"[{task_name}] Step {i + 1} Generating image with seed: {actual_seed}"
            )

            img = generate_first_image(
                model_path=out_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=512,
                height=512,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                seed=actual_seed,
            )

            if img:
                images.append((img, current_alpha))
            else:
                logging.warning(f"[{task_name}] Step {i + 1} Image generation failed.")

            # 次のイテレーションのBaseを今回の出力モデルにする
            current_base = out_path

        except Exception as e:
            logging.error(f"[{task_name}] Step {i + 1} Failed: {e}")
            break
        finally:
            if os.path.exists(tmp_cfg):
                os.remove(tmp_cfg)

    # 全ステップ終了後、グリッド画像を生成
    if images:
        w, h = 512, 512
        grid_w = len(images) * w
        grid_h = h
        grid_img = Image.new("RGB", (grid_w, grid_h))
        draw = ImageDraw.Draw(grid_img)

        for idx, (img, alpha) in enumerate(images):
            px = idx * w
            grid_img.paste(img, (px, 0))
            text = f"Step {idx + 1}: Alpha {alpha:.3f}"
            # 背景色をつけて見やすくする
            draw.rectangle([(px + 5, 5), (px + 150, 25)], fill="black")
            draw.text((px + 10, 10), text, fill="white")

        grid_out = os.path.join(output_dir, "poison_merge_grid.png")
        grid_img.save(grid_out)
        logging.info(f"[{task_name}] Grid image saved to {grid_out}")

    logging.info(f"[{task_name}] Poison Merge Finished.")
