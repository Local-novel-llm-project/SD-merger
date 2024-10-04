import argparse
import logging
import os
import shutil

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
    WeightMatchingStrategy,  # 追加
    get_preprocessing_strategy,
)
from module.preprocess_target import (
    WeightMatchingTargetStrategy,  # 追加
    get_target_preprocessing_strategy,
)
from module.utility import (
    assemble_model_from_cache,
    generate_filename,
    load_model,
    load_yaml_config,
    save_model,
)

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,  # ログレベルをDEBUGに設定
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

            # プリプロセスの適用
            if isinstance(preprocessing_strategy, WeightMatchingStrategy):
                # モデルのパスを渡す
                _, cache_dir = preprocessing_strategy.preprocess(
                    left_model_path, right_model_path, key_patterns
                )
                # キャッシュから右モデルを組み立て
                right_model = assemble_model_from_cache(cache_dir)
                # キャッシュを削除
                shutil.rmtree(cache_dir)
                # 左モデルを読み込む（必要なキーのみ）
                left_model = load_model(left_model_path)
                if key_patterns is not None:
                    left_model = {
                        k: v
                        for k, v in left_model.items()
                        if any(p in k for p in key_patterns)
                    }
            else:
                # 左右のモデルの読み込み
                left_model = load_model(left_model_path)
                right_model = load_model(right_model_path)

                # プリプロセスの適用
                left_model, right_model = preprocessing_strategy.preprocess(
                    left_model, right_model, key_patterns
                )

            # 左右のモデルをマージ
            left_right_merged_model = strategy.calculate(
                None,
                left_model,
                right_model,
                left_right_velocity,
                key_patterns,
            )

            # ターゲットモデルの処理
            if target_model_path:
                target_model = {}

                if isinstance(
                    target_preprocessing_strategy, WeightMatchingTargetStrategy
                ):
                    # ターゲットモデルのプリプロセス適用
                    target_cache_dir = target_preprocessing_strategy.preprocess(
                        target_model_path,
                        key_patterns,
                        reference_model_path=left_model_path,
                    )
                    # キャッシュからターゲットモデルを組み立て
                    target_model = assemble_model_from_cache(target_cache_dir)
                    # キャッシュを削除
                    shutil.rmtree(target_cache_dir)
                else:
                    # ターゲットモデルを読み込み
                    target_model = load_model(target_model_path)
                    if key_patterns is not None:
                        target_model = {
                            k: v
                            for k, v in target_model.items()
                            if any(p in k for p in key_patterns)
                        }
                    # プリプロセスの適用
                    target_model = target_preprocessing_strategy.preprocess(
                        target_model,
                        key_patterns,
                        reference_model=left_right_merged_model,
                    )

                task_merge = progress.add_task(
                    "[cyan]マージ中...", total=len(target_model.keys())
                )

                def update_callback():
                    progress.update(task_merge, advance=1)

                normalization_strategy.set_progress_callback(update_callback)
                with torch.no_grad():
                    target_model = SDKeyWrapper(
                        normalization_strategy.calculate(
                            target_model,
                            left_right_merged_model,
                            None,
                            target_strategy,
                            strategy,
                            left_right_velocity,
                            target_velocity,
                            key_patterns,
                        )
                    )
            else:
                # ターゲットモデルがない場合、計算結果をそのまま保存
                with torch.no_grad():
                    target_model = target_strategy.calculate(
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
    save_model(target_model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデルの差分計算とマージツール")
    parser.add_argument(
        "-c", "--config", type=str, default="sd_config.yaml", help="設定ファイルのパス"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./merged", help="出力ディレクトリのパス"
    )
    args = parser.parse_args()

    main(args.config, args.output)
