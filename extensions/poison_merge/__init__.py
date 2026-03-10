import os
import math
import logging
from pprint import pformat
import gc
import torch

from module.extension_manager import register_pre_config_hook

logger = logging.getLogger(__name__)


def get_decay_weight(
    step: int, total_steps: int, decay_type: str, initial_alpha: float = 1.0
) -> float:
    """減衰カーブに従って現在のアルファ値を計算する"""
    if total_steps <= 1:
        return initial_alpha

    progress = step / (total_steps - 1)  # 0.0 to 1.0

    if decay_type == "linear":
        return initial_alpha * (1.0 - progress)
    elif decay_type == "exponential":
        # exp(-3 * progress) will decay from 1 to ~0.05
        return initial_alpha * math.exp(-3.0 * progress)
    elif decay_type == "cosine":
        return initial_alpha * (0.5 * (1.0 + math.cos(math.pi * progress)))
    else:
        # Default to linear
        return initial_alpha * (1.0 - progress)


def run_poison_pipeline(config: dict) -> dict:
    """
    設定ファイルの poison_merge セクションを処理するフック。
    反復的に LoRA をモデルにマージし、各ステップで画像生成を行う。
    """
    poison_config = config.get("poison_merge")
    if not poison_config:
        return config

    logger.info("Poison Merge パイプラインを開始します...")
    logger.debug(f"Poison Config: {pformat(poison_config)}")

    # 依存モジュールのインポート (循環参照を避けるため遅延インポート)
    from main import main as merger_main
    from module.generation import generate_first_image
    import tempfile
    import yaml

    base_model = poison_config.get("base_model")
    lora_model = poison_config.get("lora_model")
    iterations = poison_config.get("iterations", 3)
    decay_type = poison_config.get("decay_type", "linear")
    initial_alpha = poison_config.get("initial_alpha", 1.0)
    output_dir = poison_config.get("output_dir", "./models/output/poison_merge")
    prompt = poison_config.get("prompt", "")
    negative_prompt = poison_config.get("negative_prompt", "")
    seed = poison_config.get("seed", -1)

    os.makedirs(output_dir, exist_ok=True)

    current_base = base_model
    last_output_path = None

    for step in range(iterations):
        alpha = get_decay_weight(step, iterations, decay_type, initial_alpha)
        logger.info(
            f"--- Iteration {step + 1}/{iterations} (decay={decay_type}, alpha={alpha:.4f}) ---"
        )

        step_output_name = f"poison_step_{step + 1}.safetensors"
        step_output_path = os.path.join(output_dir, step_output_name)

        # 1. マージ用の一時 Config を作成し、merger_main を呼び出してマージ実行
        step_config = {
            "target_model": current_base,
            "models": [
                {
                    "left": current_base,
                    "right": lora_model,
                    "strategy": "addition",  # LoRAを単純加算
                    "velocity": alpha,
                    "key_patterns": ["."],
                }
            ],
            "output_name": step_output_name,
        }

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
            yaml.dump(step_config, f)
            tmp_cfg_path = f.name

        logger.info(
            f"マージ実行中: {current_base} + {lora_model} (alpha={alpha:.4f}) -> {step_output_path}"
        )
        merger_main(tmp_cfg_path, output_dir)
        os.remove(tmp_cfg_path)

        if not os.path.exists(step_output_path):
            raise RuntimeError(
                f"マージの出力ファイルが見つかりません: {step_output_path}"
            )

        # 2. 画像の生成
        if prompt:
            logger.info("サンプル画像を生成中...")
            img = generate_first_image(
                model_path=step_output_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=512,
                height=512,
                steps=20,
                cfg=7.0,
            )

            if img:
                img_path = os.path.join(output_dir, f"poison_step_{step + 1}.png")
                img.save(img_path)
                logger.info(f"画像を保存しました: {img_path}")
            else:
                logger.warning("画像の生成に失敗しました。")

        # 3. 次のステップに向けて Base Model を更新
        current_base = step_output_path
        last_output_path = step_output_path

        # メモリ解放
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Poison Merge パイプラインが完了しました。")

    config["_skip_merge"] = True
    config["_skip_merge_output"] = last_output_path or current_base

    return config


def setup():
    register_pre_config_hook(run_poison_pipeline)
