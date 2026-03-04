import os
import sys
import json
import logging
import argparse

import torch
import sd_mecha
from rich.console import Console

from module.logging_config import logger
from module.exceptions import ConfigError, SDMergerError
from module.config_schema import MergeConfig
from pydantic import ValidationError

from module.calc_method import get_calculation_strategy
from module.calc_target import (
    get_normalization_calculation_strategy,
    get_target_calculation_strategy,
)
from module.utility import generate_filename, load_yaml_config
from module.extension_manager import (
    load_extensions,
    run_pre_config_hooks,
    run_pre_merge_hooks,
    run_post_merge_hooks,
)

# logger configuration is handled by module.logging_config

console = Console()


@sd_mecha.merge_method
def scale_tensor(
    a: sd_mecha.Parameter(torch.Tensor), scale: sd_mecha.Parameter(torch.Tensor) = 1.0
) -> sd_mecha.Return(torch.Tensor):
    return a * scale


def main(config_path: str, output_dir: str) -> None:
    """メイン処理。設定ファイルに従いモデルのマージを sd-mecha レシピとして構築して実行する。

    Args:
        config_path: YAML 設定ファイルのパス。
        output_dir: 出力ディレクトリのパス。
    """
    # 設定を読み込み、拡張機能による事前加工（MBWの解決など）を行う
    raw_config = load_yaml_config(config_path)
    try:
        validated_config = MergeConfig(**raw_config)
        # We process extensions based on raw config for now, then can re-validate or just use dict as before.
        # But let's work with the validated dict
        config = run_pre_config_hooks(validated_config.model_dump())
    except ValidationError as e:
        logger.error(f"コンフィグのバリデーションエラー: {e}")
        raise ConfigError("Invalid configuration syntax or types.", original_error=e)

    target_model_path = config.get("target_model")
    if target_model_path:
        recipe = sd_mecha.model(target_model_path)
    else:
        recipe = None

    models = config.get("models", [])
    if not models:
        logger.error("設定ファイルにモデルが指定されていません。")
        return

    # SDXL キーの自動変換などの機能は sd-mecha がモデルコンフィグを自動推論して適用するため
    # 以前のような use_sdxl_keys フラグの手動管理は基本不要になります。

    for model_config in models:
        left_node = sd_mecha.model(model_config["left"])
        right_node = sd_mecha.model(model_config["right"])
        target_velocity = model_config.get("velocity", 1.0)
        left_right_velocity = model_config.get("left_right_velocity", 1.0)
        strategy_name = model_config.get("strategy", "addition")
        key_patterns = model_config.get("key_patterns", None)
        replace_with = model_config.get("replace_with", None)
        target_strategy_name = model_config.get("target_strategy", "addition")
        normalization_strategy_name = model_config.get("normalization_strategy", "none")

        if not key_patterns:
            if recipe is None:
                logger.error("target_model と key_patterns の両方が未指定です。どちらかを指定してください。")
                raise ConfigError("target_model と key_patterns の両方が未指定です。")
            # 現在は sd-mecha が全キーを走査するため、必要であれば事前検出などは別に行う必要があります。
            # プロジェクトの互換性維持のためここは一度エラーにします。
            logger.error('key_patterns の指定は必須です。(全キーを指定する場合は "." 等を指定)')
            raise ConfigError("key_patterns の指定は必須です。")

        calc_func = get_calculation_strategy(strategy_name, replace_with)
        target_func = get_target_calculation_strategy(target_strategy_name)
        norm_func = get_normalization_calculation_strategy(normalization_strategy_name)

        patterns_json = json.dumps(key_patterns)

        # left/right の差分計算ノード
        diff_node = calc_func(
            left_node,
            right_node,
            velocity=left_right_velocity,
            key_patterns_json=patterns_json,
        )

        if recipe is not None:
            if target_strategy_name == "angle":
                # angle は特殊で diff_l, diff_r を求める必要がある
                diff_l = calc_func(
                    left_node,
                    recipe,
                    velocity=left_right_velocity,
                    key_patterns_json=patterns_json,
                )
                diff_r = calc_func(
                    right_node,
                    recipe,
                    velocity=left_right_velocity,
                    key_patterns_json=patterns_json,
                )
                merged = target_func(
                    recipe,
                    diff_l=diff_l,
                    diff_r=diff_r,
                    left=left_node,
                    right=right_node,
                    key_patterns_json=patterns_json,
                )
            else:
                merged = target_func(
                    recipe,
                    diff_node,
                    velocity=target_velocity,
                    key_patterns_json=patterns_json,
                )

            if norm_func is not None:
                merged = norm_func(recipe, merged)

            recipe = merged
        else:
            # ターゲットモデルがない場合、left/right の計算結果をそのまま使用
            recipe = scale_tensor(diff_node, scale=target_velocity)

    # 出力ファイル名の決定 (設定があればそれを優先)
    output_filename = config.get("output_name")
    if not output_filename:
        # 全モデル設定から代表名を取得してファイル名を生成
        first_left_name = os.path.basename(models[0]["left"])
        last_right_name = os.path.basename(models[-1]["right"])
        output_filename = generate_filename(first_left_name, last_right_name)

    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    recipe = run_pre_merge_hooks(config, recipe)

    logger.info(f"マージ処理を実行し、{output_path} に保存します...")
    logger.info("sd-mecha がストリーミング処理を開始します。")
    # sd-mecha によるストリーミングマージの実行
    sd_mecha.set_log_level(logging.INFO)
    try:
        sd_mecha.merge(recipe, output=output_path)
    except Exception as e:
        logger.error(f"sd-mecha merging error: {e}")
        from module.exceptions import MergeError

        raise MergeError("Merge failed during sd_mecha processing", original_error=e)

    logger.info("マージが完了しました。")

    # post_merge_hookの実行
    run_post_merge_hooks(config, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデルの差分計算とマージツール (sd-mecha版)")
    parser.add_argument("-c", "--config", type=str, default="sd_config.yaml", help="設定ファイルのパス")
    parser.add_argument("-o", "--output", type=str, default="./merged", help="出力ディレクトリのパス")
    parser.add_argument("-d", "--debug", action="store_true", help="DEBUGログレベルを有効にする")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        sd_mecha.set_log_level(logging.DEBUG)

    # 拡張機能の読み込み
    load_extensions()

    try:
        main(args.config, args.output)
    except SDMergerError as e:
        logger.error(f"Application Error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error occurred.")
        sys.exit(1)
