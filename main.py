"""SD モデルの差分計算とマージツール。

YAML 設定ファイルに基づき、複数の Stable Diffusion モデル間の
差分計算・マージを行い、結果を safetensors 形式で保存する。
"""

import os
import sys

import torch
import logging
import argparse

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from module.calc_method import get_calculation_strategy
from module.calc_target import (
    get_normalization_calculation_strategy,
    get_target_calculation_strategy,
)
from module.const import SDKeyWrapper
from module.utility import (
    generate_filename,
    load_model,
    load_yaml_config,
    save_model,
)

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()],
)

console = Console()


def main(config_path: str, output_dir: str) -> None:
    """メイン処理。設定ファイルに従いモデルのマージを実行する。

    Args:
        config_path: YAML 設定ファイルのパス。
        output_dir: 出力ディレクトリのパス。
    """
    config = load_yaml_config(config_path)

    target_model_path = config.get("target_model")
    if target_model_path:
        target_model = load_model(target_model_path)
    else:
        target_model = None

    models = config.get("models", [])
    if not models:
        logging.error("設定ファイルにモデルが指定されていません。")
        return

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]モデルを処理中...", total=len(models))

        for model_config in models:
            # 必須フィールドのバリデーション
            required_fields = ["left", "right", "velocity", "strategy"]
            missing = [f for f in required_fields if f not in model_config]
            if missing:
                logging.error(f"モデル設定に必須フィールドが不足しています: {missing}")
                sys.exit(1)

            left_model_path = model_config["left"]
            right_model_path = model_config["right"]
            target_velocity = model_config["velocity"]
            left_right_velocity = model_config.get("left_right_velocity", 1.0)
            strategy_name = model_config["strategy"]
            key_patterns = model_config.get("key_patterns", None)
            replace_with = model_config.get("replace_with", None)
            target_strategy_name = model_config.get("target_strategy", "addition")
            normalization_strategy_name = model_config.get("normalization_strategy", "none")

            strategy = get_calculation_strategy(strategy_name, replace_with)
            target_strategy = get_target_calculation_strategy(target_strategy_name)
            normalization_strategy = get_normalization_calculation_strategy(normalization_strategy_name)

            # SDXL キー判定: target_model があればその設定を使用、なければ True をデフォルトに
            use_sdxl_keys = target_model.use_sdxl_keys if target_model is not None else True

            if not key_patterns:
                if target_model is None:
                    logging.error("target_model と key_patterns の両方が未指定です。どちらかを指定してください。")
                    sys.exit(1)
                keys_output_path = os.path.join(output_dir, "available_keys.txt")
                os.makedirs(output_dir, exist_ok=True)
                with open(keys_output_path, "w", encoding="utf-8") as f:
                    f.write("追加可能なキー:\n")
                    for key in target_model.keys():
                        f.write(f"{key}: {list(target_model[key].shape)}\n")
                logging.info(
                    f"設定ファイルにキーのパターンが指定されていません。"
                    f"追加可能なキーが {keys_output_path} に書き出されました。終了します。"
                )
                sys.exit(1)

            left_model = load_model(left_model_path, use_sdxl_keys=use_sdxl_keys)
            right_model = load_model(right_model_path, use_sdxl_keys=use_sdxl_keys)

            if target_model is not None:
                # 対象レイヤーに一致するキー数で進捗バーを設定
                matching_keys = [k for k in target_model.keys() if any(pat in k for pat in key_patterns)]
                task_merge = progress.add_task(
                    "[cyan]マージ中...", total=len(matching_keys) if matching_keys else len(target_model.keys())
                )

                def update_callback():
                    progress.update(task_merge, advance=1)

                normalization_strategy.set_progress_callback(update_callback)
                with torch.no_grad():
                    final_model = SDKeyWrapper(
                        normalization_strategy.calculate(
                            target_model,
                            left_model,
                            right_model,
                            target_strategy,
                            strategy,
                            left_right_velocity,
                            target_velocity,
                            key_patterns,
                        ),
                        use_sdxl_keys=use_sdxl_keys,
                    )
            else:
                # ターゲットモデルがない場合、left/right の計算結果をそのまま使用
                with torch.no_grad():
                    result = strategy.calculate(
                        left_model,
                        right_model,
                        left_right_velocity,
                        key_patterns,
                    )
                    # target_velocity を in-place で適用（メモリ効率向上）
                    for v in result.values():
                        v.mul_(target_velocity)
                    final_model = SDKeyWrapper(result, use_sdxl_keys=use_sdxl_keys)

            del left_model, right_model
            # GPU メモリの明示的な解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            progress.update(task, advance=1)

    # 全モデル設定から代表名を取得してファイル名を生成
    first_left_name = os.path.basename(models[0]["left"])
    last_right_name = os.path.basename(models[-1]["right"])
    output_filename = generate_filename(first_left_name, last_right_name)
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    save_model(final_model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデルの差分計算とマージツール")
    parser.add_argument("-c", "--config", type=str, default="sd_config.yaml", help="設定ファイルのパス")
    parser.add_argument("-o", "--output", type=str, default="./merged", help="出力ディレクトリのパス")
    parser.add_argument("-d", "--debug", action="store_true", help="DEBUGログレベルを有効にする")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args.config, args.output)
