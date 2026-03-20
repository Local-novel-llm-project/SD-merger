import os
import math
import logging
from typing import Any, Dict, List


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


def normalize_lora_models(poison_config: Dict[str, Any]) -> List[str]:
    """poison_merge 設定から LoRA モデル一覧を抽出する。"""
    candidates = poison_config.get("lora_models")
    if not candidates:
        candidates = poison_config.get("lora_model")

    if candidates is None:
        return []

    if isinstance(candidates, str):
        candidates = [candidates]
    elif not isinstance(candidates, list):
        candidates = list(candidates)

    normalized = [str(item).strip() for item in candidates if str(item).strip()]
    return normalized


def build_iteration_plan(
    current_base: str,
    lora_models: List[str],
    output_dir: str,
    iteration: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    """1 イテレーション分の LoRA 適用計画を組み立てる。"""
    if not lora_models:
        return []

    output_name = f"poison_step_{iteration}_alpha_{alpha:.2f}.safetensors"
    output_path = os.path.join(output_dir, output_name)

    return [
        {
            "left": current_base,
            "models": list(lora_models),
            "ratios": [float(alpha)] * len(lora_models),
            "alpha": float(alpha),
            "output_name": output_name,
            "output_path": output_path,
        }
    ]


def resolve_lora_merge_precision(dtype_name: str | None) -> str:
    """sd-merger の dtype 名を kohya の precision 名へ変換する。"""
    precision_map = {
        "float16": "fp16",
        "float32": "float",
        "bfloat16": "bf16",
    }
    return precision_map.get(str(dtype_name or "float32"), "float")


def infer_model_is_sdxl(model_path: str) -> bool:
    """ベース checkpoint のキー構造から SDXL かどうかを判定する。"""
    from module.utility import load_model

    wrapper = load_model(model_path, lazy_load=True)
    return bool(getattr(wrapper, "is_xl", False))


def build_poison_lora_apply_operation(
    base_model: str,
    lora_models: List[str],
    ratios: List[float],
    output_path: str,
    *,
    sdxl: bool,
    precision: str,
    save_precision: str,
    v2: bool = False,
    no_metadata: bool = False,
) -> Dict[str, Any]:
    """Poison Merge の 1 イテレーション分を lora_ops apply 用オペレーションへ変換する。"""
    return {
        "type": "apply",
        "sd_model": base_model,
        "models": list(lora_models),
        "ratios": [float(ratio) for ratio in ratios],
        "output": output_path,
        "sdxl": bool(sdxl),
        "precision": precision,
        "save_precision": save_precision,
        "v2": bool(v2),
        "no_metadata": bool(no_metadata),
    }


def apply_lora_stage(
    base_model: str,
    lora_models: List[str],
    ratios: List[float],
    output_path: str,
    *,
    sdxl: bool,
    precision: str,
    save_precision: str,
    v2: bool = False,
    no_metadata: bool = False,
) -> str:
    """既存の lora_ops 実装を使って checkpoint へ LoRA 群を一括適用する。"""
    from extensions import lora_ops

    operation = build_poison_lora_apply_operation(
        base_model,
        lora_models,
        ratios,
        output_path,
        sdxl=sdxl,
        precision=precision,
        save_precision=save_precision,
        v2=v2,
        no_metadata=no_metadata,
    )
    return lora_ops._run_merge_lora(operation)


def run_poison_merge(
    config: Dict[str, Any], task_name: str = "Poison Merge"
) -> str | None:
    """Poison Merge のメインイテレーションを実行する。

    各イテレーションで以下の処理を行う:
    1. Base モデルへ LoRA を適用
    2. 画像を生成して保存
    3. 出力されたマージ済みモデルを次ステップの Base として続行
    """
    from module.generation import generate_first_image
    from PIL import Image, ImageDraw

    p_config = config["poison_merge"]
    base_model = p_config["base_model"]
    lora_models = normalize_lora_models(p_config)
    iterations = p_config.get("iterations", 3)
    decay_type = p_config.get("decay_type", "linear")
    initial_alpha = p_config.get("initial_alpha", 1.0)
    alpha_overrides_str = p_config.get("alpha_overrides", "")
    output_dir = p_config["output_dir"]
    prompt = p_config.get("prompt", "A beautiful landscape")
    negative_prompt = p_config.get("negative_prompt", "blurry")
    seed = p_config.get("seed", -1)

    precision = p_config.get("precision") or resolve_lora_merge_precision(config.get("dtype"))
    save_precision = p_config.get("save_precision") or precision
    if "sdxl" in p_config:
        is_sdxl = bool(p_config.get("sdxl"))
    else:
        is_sdxl = infer_model_is_sdxl(base_model)
    v2 = bool(p_config.get("v2", False))
    no_metadata = bool(p_config.get("no_metadata", False))

    alphas = calculate_alphas(initial_alpha, iterations, decay_type)
    if not lora_models:
        raise ValueError("Poison Merge requires at least one LoRA model.")

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
    logging.info(f"[{task_name}] LoRAs: {lora_models}")
    logging.info(
        f"[{task_name}] LoRA apply mode: family={'SDXL' if is_sdxl else 'SD1/SD2'}, "
        f"precision={precision}, save_precision={save_precision}"
    )

    os.makedirs(output_dir, exist_ok=True)
    images = []

    current_base = base_model
    last_output_path = None
    failure: Exception | None = None

    for i, current_alpha in enumerate(alphas):
        logging.info(
            f"[{task_name}] Step {i + 1}/{iterations}, Alpha: {current_alpha:.3f}"
        )

        iteration_plan = build_iteration_plan(
            current_base,
            lora_models,
            output_dir,
            i + 1,
            current_alpha,
        )

        try:
            for stage_index, stage in enumerate(iteration_plan, start=1):
                logging.info(
                    f"[{task_name}] Step {i + 1}.{stage_index}/{len(iteration_plan)} "
                    f"Applying {len(stage['models'])} LoRAs to base checkpoint."
                )
                apply_lora_stage(
                    stage["left"],
                    stage["models"],
                    stage["ratios"],
                    stage["output_path"],
                    sdxl=is_sdxl,
                    precision=precision,
                    save_precision=save_precision,
                    v2=v2,
                    no_metadata=no_metadata,
                )

                if not os.path.exists(stage["output_path"]):
                    raise FileNotFoundError(
                        f"Poison Merge output not found: {stage['output_path']}"
                    )

                current_base = stage["output_path"]
                last_output_path = stage["output_path"]

            logging.info(
                f"[{task_name}] Step {i + 1} Merge completed: {os.path.basename(current_base)}"
            )

            # 画像生成
            import random

            actual_seed = (
                int(seed) if int(seed) > 0 else random.randint(1, 1125899906842624)
            )
            logging.info(
                f"[{task_name}] Step {i + 1} Generating image with seed: {actual_seed}"
            )

            img = generate_first_image(
                model_path=current_base,
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

        except Exception as e:
            logging.error(f"[{task_name}] Step {i + 1} Failed: {e}")
            failure = e
            break

    if failure is not None:
        raise RuntimeError(f"{task_name} failed before completion.") from failure

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
    return last_output_path
