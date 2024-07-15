import os
import torch

import logging
from rich.logging import RichHandler
from rich.progress import Progress
from rich.console import Console
import argparse


from module.calc_metod import get_calculation_strategy
from module.calc_target import (
    get_normalization_calculation_strategy,
    get_target_calculation_strategy,
)
from module.const import SDKeyWrapper
from module.utility import generate_filename, load_model, load_yaml_config, save_model

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
    target_model = (
        load_model(f"{target_model_path}.safetensors") if target_model_path else None
    )

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]モデルを処理中...", total=len(config["models"]))

        for model_config in config["models"]:
            left_model_path = f"{model_config['left']}.safetensors"
            right_model_path = f"{model_config['right']}.safetensors"
            target_velocity = model_config["velocity"]
            left_right_velocity = model_config.get("left_right_velocity", 1.0)
            strategy_name = model_config["strategy"]
            key_patterns = model_config.get("key_patterns", None)
            replace_with = model_config.get("replace_with", None)
            target_strategy_name = model_config.get("target_strategy", "addition")
            normalization_strategy_name = model_config.get(
                "normalization_strategy", "none"
            )

            strategy = get_calculation_strategy(strategy_name, replace_with)
            target_strategy = get_target_calculation_strategy(target_strategy_name)
            normalization_strategy = get_normalization_calculation_strategy(
                normalization_strategy_name
            )

            # target_model が None の場合のデフォルト値を設定
            use_sdxl_keys = target_model.use_sdxl_keys if target_model else True

            if key_patterns is None:
                with open("available_keys.txt", "w") as f:
                    f.write("追加可能なキー:\n")
                    # for key in model_diff.keys():
                    for key in target_model.keys():
                        f.write(f"{key}: {list(target_model[key].shape)}\n")
                logging.info(
                    "設定ファイルにキーのパターンが指定されていません。追加可能なキーが available_keys.txt に書き出されました。終了します。"
                )
                exit()

            left_model = load_model(left_model_path, use_sdxl_keys=use_sdxl_keys)
            right_model = load_model(right_model_path, use_sdxl_keys=use_sdxl_keys)

            if target_model:
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
                            left_model,
                            right_model,
                            target_strategy,
                            strategy,
                            left_right_velocity,
                            target_velocity,
                            key_patterns,
                        )
                    )
            else:
                # ターゲットモデルがない場合、計算結果をそのまま保存
                # target_model = model_diff
                with torch.no_grad():
                    target_model = target_strategy.calculate(
                        target_model,
                        left_model,
                        right_model,
                        strategy,
                        left_right_velocity,
                        target_velocity,
                        key_patterns,
                    )
                pass

            del left_model, right_model

            progress.update(task, advance=1)

    left_model_name = os.path.basename(model_config["left"])
    right_model_name = os.path.basename(model_config["right"])
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
