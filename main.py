import argparse
import logging
import os

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from module.calc_metod import get_calculation_strategy
from module.calc_target import (
    get_normalization_calculation_strategy,
    get_target_calculation_strategy,
)
from module.const import SDKeyWrapper
from module.preprocess_method import (
    get_preprocessing_strategy,
)
from module.preprocess_target import (
    get_target_preprocessing_strategy,
)
from module.utility import (
    generate_filename,
    load_model,
    load_processed_keys,  # 追加
    load_yaml_config,
    save_model,
    save_processed_key,  # 追加
)

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()],
)

console = Console()


def main(config_path: str, output_dir: str):
    config = load_yaml_config(config_path)

    target_model_path = config.get("target_model")

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]モデルを処理中...", total=len(config["models"]))

        for model_config in config["models"]:
            left_model_path = f"{model_config['left']}"
            right_model_path = f"{model_config['right']}"
            target_velocity = model_config["velocity"]
            left_right_velocity = model_config.get("left_right_velocity", 1.0)
            strategy_name = model_config["strategy"]
            key_patterns = model_config.get("key_patterns", None)
            replace_with = model_config.get("replace_with", None)
            target_strategy_name = model_config.get("target_strategy", "addition")
            normalization_strategy_name = model_config.get(
                "normalization_strategy", "none"
            )
            preprocessing_strategy_name = model_config.get(
                "preprocessing_strategy", "noop"
            )
            target_preprocessing_strategy_name = model_config.get(
                "target_preprocessing_strategy", "noop"
            )

            strategy = get_calculation_strategy(strategy_name, replace_with)
            target_strategy = get_target_calculation_strategy(target_strategy_name)
            normalization_strategy = get_normalization_calculation_strategy(
                normalization_strategy_name
            )
            preprocessing_strategy = get_preprocessing_strategy(
                preprocessing_strategy_name
            )
            target_preprocessing_strategy = get_target_preprocessing_strategy(
                target_preprocessing_strategy_name
            )

            # モデルの読み込みとパラメータの分離
            left_model = load_model(left_model_path)
            right_model = load_model(right_model_path)

            if key_patterns is not None:
                # key_patterns にマッチするパラメータ
                left_model_to_process = {
                    k: v
                    for k, v in left_model.items()
                    if any(p in k for p in key_patterns)
                }
                left_model_unprocessed = {
                    k: v
                    for k, v in left_model.items()
                    if not any(p in k for p in key_patterns)
                }
                right_model_to_process = {
                    k: v
                    for k, v in right_model.items()
                    if any(p in k for p in key_patterns)
                }
                right_model_unprocessed = {
                    k: v
                    for k, v in right_model.items()
                    if not any(p in k for p in key_patterns)
                }
            else:
                left_model_to_process = left_model
                left_model_unprocessed = {}
                right_model_to_process = right_model
                right_model_unprocessed = {}

            # キャッシュディレクトリの設定
            cache_dir = os.path.join(
                "cache",
                os.path.splitext(os.path.basename(left_model_path))[0]
                + "_"
                + os.path.splitext(os.path.basename(right_model_path))[0],
            )

            # キャッシュから処理済みのキーを読み込む
            processed_keys = load_processed_keys(cache_dir)

            # 未処理のキーのみを対象とする
            left_model_to_process = {
                k: v
                for k, v in left_model_to_process.items()
                if k not in processed_keys
            }
            right_model_to_process = {
                k: v
                for k, v in right_model_to_process.items()
                if k not in processed_keys
            }

            # プリプロセスの適用
            left_processed, right_processed = preprocessing_strategy.preprocess(
                left_model_to_process,
                right_model_to_process,
                key_patterns,
            )

            # 処理済みのキーをキャッシュに保存
            for key in left_processed.keys():
                save_processed_key(cache_dir, key, left_processed[key])

            # キャッシュから読み込んだキーをマージ
            left_processed.update(processed_keys)
            right_processed.update(
                {
                    key: processed_keys[key]
                    for key in processed_keys
                    if key in right_model_to_process
                }
            )

            # 左右のモデルをマージ
            left_right_merged_model_processed = strategy.calculate(
                None,
                left_processed,
                right_processed,
                left_right_velocity,
                key_patterns,
            )

            # 処理済みパラメータと未処理パラメータの統合
            left_right_merged_model = {
                **left_right_merged_model_processed,
                **left_model_unprocessed,
            }

            # ターゲットモデルの処理
            if target_model_path:
                target_model = load_model(target_model_path)

                if key_patterns is not None:
                    target_model_to_process = {
                        k: v
                        for k, v in target_model.items()
                        if any(p in k for p in key_patterns)
                    }
                    target_model_unprocessed = {
                        k: v
                        for k, v in target_model.items()
                        if not any(p in k for p in key_patterns)
                    }
                else:
                    target_model_to_process = target_model
                    target_model_unprocessed = {}

                # ターゲットモデルのプリプロセス適用
                target_processed = target_preprocessing_strategy.preprocess(
                    target_model_to_process,
                    key_patterns,
                    reference_model=left_right_merged_model_processed,
                )

                task_merge = progress.add_task(
                    "[cyan]マージ中...",
                    total=len(target_processed.keys()),
                )

                def update_callback():
                    progress.update(task_merge, advance=1)

                normalization_strategy.set_progress_callback(update_callback)
                with torch.no_grad():
                    target_model_processed = SDKeyWrapper(
                        normalization_strategy.calculate(
                            target_processed,
                            left_right_merged_model_processed,
                            right_model_to_process,
                            target_strategy,
                            strategy,
                            left_right_velocity,
                            target_velocity,
                            key_patterns,
                            left_model=left_model_to_process,
                        )
                    )

                # 処理済みパラメータと未処理パラメータの統合
                final_model = {
                    **target_model_processed,
                    **target_model_unprocessed,
                }
            else:
                # ターゲットモデルがない場合、計算結果をそのまま保存
                with torch.no_grad():
                    final_model = target_strategy.calculate(
                        None,
                        left_right_merged_model,
                        None,
                        strategy,
                        left_right_velocity,
                        target_velocity,
                        key_patterns,
                    )

            del left_model, right_model

            progress.update(task, advance=1)

    left_model_name = os.path.basename(config["models"][-1]["left"])
    right_model_name = os.path.basename(config["models"][-1]["right"])
    output_filename = generate_filename(left_model_name, right_model_name)
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    save_model(final_model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデルの差分計算とマージツール")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="sd_config.yaml",
        help="設定ファイルのパス",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./merged",
        help="出力ディレクトリのパス",
    )
    args = parser.parse_args()

    main(args.config, args.output)
