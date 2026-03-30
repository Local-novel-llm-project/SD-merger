"""SD モデルの left/right 間の差分計算ストラテジーモジュール (sd-mecha 版)。

sd-mecha の @merge_method デコレータを使用し、
減算、加算、乗算、平均、混合、置き換えの各処理をノードとして定義する。
"""

import json
from typing import Optional
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

from module.extension_manager import get_extension_strategies

_CACHE_PATTERNS = {}


def _is_key_matched(key: str, key_patterns_json: str) -> bool:
    if not key_patterns_json or key_patterns_json == "[]":
        return True
    if key_patterns_json not in _CACHE_PATTERNS:
        _CACHE_PATTERNS[key_patterns_json] = json.loads(key_patterns_json)
    patterns = _CACHE_PATTERNS[key_patterns_json]
    return any(p in key for p in patterns)


@merge_method
def subtract_velocity(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """減算ストラテジー: (a - b) * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return torch.zeros_like(a)
    return (a - b) * velocity


@merge_method
def add_velocity(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """加算ストラテジー: (a + b) * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return torch.zeros_like(a)
    return (a + b) * velocity


@merge_method
def multiply_velocity(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """乗算ストラテジー: (a * b) * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return torch.zeros_like(a)
    return (a * b) * velocity


@merge_method
def average_velocity(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """平均ストラテジー: (a + b) / 2 * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return torch.zeros_like(a)
    return (a + b) / 2.0 * velocity


@merge_method
def mix_velocity(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """混合ストラテジー: a * (1 - velocity) + b * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return torch.zeros_like(a)
    return a * (1.0 - velocity) + b * velocity


@merge_method
def replace_left_velocity(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """置き換えストラテジー(left): a * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return a  # 置換対象外はそのまま a を返す
    return a * velocity


@merge_method
def replace_right_velocity(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """置き換えストラテジー(right): b * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return b  # 置換対象外はそのまま b を返す
    return b * velocity


def get_calculation_strategy(strategy_name: str, replace_with: Optional[str] = None):
    """名前から計算ストラテジーメソッド(Callable)を取得する。

    Args:
        strategy_name: ストラテジー名
            ('subtraction', 'addition', 'multiplication', 'average', 'mix', 'replace')。
        replace_with: 'replace' ストラテジー用の置き換え元 ('left' or 'right')。

    Returns:
        対応する sd-mecha の merge_method。

    Raises:
        ValueError: 未知のストラテジー名が指定された場合。
    """
    strategies = {
        "subtraction": subtract_velocity,
        "addition": add_velocity,
        "multiplication": multiply_velocity,
        "average": average_velocity,
        "mix": mix_velocity,
        "replace": replace_left_velocity if replace_with == "left" else replace_right_velocity,
    }

    # 拡張機能から登録された戦略を適用
    ext_strats = get_extension_strategies()
    strategies.update(ext_strats)

    if strategy_name == "replace" and replace_with not in ("left", "right"):
        raise ValueError("'replace' ストラテジーには replace_with ('left' or 'right') の指定が必須です。")

    if strategy_name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"未知の計算方式: {strategy_name}。利用可能: {available}")

    return strategies[strategy_name]
