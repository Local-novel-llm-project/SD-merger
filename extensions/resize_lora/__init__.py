import os
import logging
from typing import List, Dict

from module.extension_manager import register_pre_config_hook


def run_resize_lora(config: dict) -> dict:
    """
    設定ファイルの resize_lora セクションを処理するフック。
    この機能は SDXL に特化しています。
    """
    resize_ops = config.get("resize_lora")
    if not resize_ops:
        return config

    logging.info("LoRAリサイズ(SDXL専用)の操作を開始します...")

    # resize_lora本体のロードパス追加
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        from .resize_lora import ResizeRecipe, load_lora_or_merge, process_lora_model
        import torch
        from .loralib import BaseCheckpoint, JsonCache, PairedLoraModel

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 複数操作(operations)のリストとして処理する
        operations = resize_ops.get("operations", [])
        if not operations and "base_model" in resize_ops:
            # 古い単一操作フォーマットへの対応
            operations = [resize_ops]

        for op in operations:
            base_model_path = op.get("base_model")
            target_loras = op.get(
                "target_loras", []
            )  # 単独のLoRAパスやマージ構文(lora1:0.7,lora2:0.3)のリスト
            output_folder = op.get("output_folder", "./")
            recipes_str = op.get("recipes", "fro_ckpt=1,thr=-3.5")
            output_dtype_str = str(op.get("output_dtype", "16"))

            output_dtype = torch.float16 if output_dtype_str == "16" else torch.float32

            if not base_model_path or not target_loras:
                logging.warning(
                    "resize_lora: 'base_model' または 'target_loras' が指定されていません。スキップします。"
                )
                continue

            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            # キャッシュファイルのロード
            norms_cache = JsonCache("norms_cache.json")

            logging.info(f"Base Checkpoint '{base_model_path}' を読み込んでいます...")
            checkpoint = BaseCheckpoint(base_model_path, cache=norms_cache)

            # レシピの解析
            score_recipes = []
            for recipe_str in recipes_str.split(":"):
                try:
                    score_recipes.append(ResizeRecipe(recipe_str))
                except Exception as e:
                    logging.error(f"レシピ '{recipe_str}' の解析エラー: {e}")

            if not score_recipes:
                logging.error("有効なレシピがありません。スキップします。")
                continue

            # 各LoRAに対して処理を実行
            for lora_path in target_loras:
                logging.info(f"LoRAモデルを処理中: {lora_path}")
                try:
                    lora_dict = load_lora_or_merge(
                        lora_path, device=device, dtype=torch.float32
                    )
                    paired = PairedLoraModel(lora_dict, checkpoint)

                    process_lora_model(
                        lora_model=paired,
                        recipes=score_recipes,
                        output_folder=output_folder,
                        output_dtype=output_dtype,
                        device=device,
                    )
                except Exception as e:
                    logging.error(f"'{lora_path}' の処理中にエラーが発生しました: {e}")

            # キャッシュを保存
            norms_cache.save(discard=True)

    except ImportError as e:
        logging.error(
            f"resize_lora機能のロードに失敗しました。loralib等が正しく配置されているか確認してください: {e}"
        )

    # resize_lora処理のみを行う場合は、後続のモデルマージを行わないようにする
    if config.get("models") and resize_ops.get("stop_after_resize", True):
        logging.info(
            "resize_loraが完了しました。stop_after_resizeがTrueのため、マージ処理をスキップします。"
        )
        config["models"] = []

    return config


def setup():
    register_pre_config_hook(run_resize_lora)
