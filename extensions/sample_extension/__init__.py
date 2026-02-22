import logging
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
from module.extension_manager import (
    register_strategy,
    register_pre_merge_hook,
    register_post_merge_hook,
)


@merge_method
def custom_sample_strategy(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """
    サンプル拡張機能によって追加されたカスタム計算戦略。
    A と B の Tensor の最大値を取って減衰させる特殊なテスト用ロジック。
    """
    # 実際にはここに高度な計算（コサイン類似度やノイズ付与など）を組み込めます。
    return torch.maximum(a, b) * velocity * 0.9


def my_pre_merge_hook(config: dict, recipe):
    logging.info("[Sample Extension] Pre-merge フックが呼び出されました。")
    return recipe


def my_post_merge_hook(config: dict, output_path: str):
    logging.info(f"[Sample Extension] Post-merge フックが呼び出されました。マージ出力先: {output_path}")


def setup():
    """拡張機能マネージャから最初に呼ばれる初期化関数"""
    logging.info("Sample Extension の初期化を開始します。")

    # カスタム計算戦略を "sample_max" という名前で登録する
    register_strategy("sample_max", custom_sample_strategy)

    # レシピ構築前後で実行されるフックを登録する
    register_pre_merge_hook(my_pre_merge_hook)
    register_post_merge_hook(my_post_merge_hook)

    logging.info("Sample Extension の初期化が完了しました。")
