"""SD モデルのターゲットモデルに対する差分適用ストラテジーモジュール (sd-mecha 版)。

target_model に diff の適用と正規化を行うロジックを sd-mecha ノードとして提供する。
"""

import json
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

_CACHE_PATTERNS = {}


def _is_key_matched(key: str, key_patterns_json: str) -> bool:
    if not key_patterns_json or key_patterns_json == "[]":
        return True
    if key_patterns_json not in _CACHE_PATTERNS:
        _CACHE_PATTERNS[key_patterns_json] = json.loads(key_patterns_json)
    patterns = _CACHE_PATTERNS[key_patterns_json]
    return any(p in key for p in patterns)


# --- Target Calculation Strategies ---


@merge_method
def target_addition(
    target: Parameter(Tensor),
    diff: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """加算ストラテジー: target += diff * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return target
    return target + diff * velocity


@merge_method
def target_subtraction(
    target: Parameter(Tensor),
    diff: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """減算ストラテジー: target -= diff * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return target
    return target - diff * velocity


@merge_method
def target_multiplication(
    target: Parameter(Tensor),
    diff: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """乗算ストラテジー: target *= diff * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return target
    return target * diff * velocity


@merge_method
def target_mix(
    target: Parameter(Tensor),
    diff: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """混合ストラテジー: target * (1 - velocity) + diff * velocity"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return target
    return target * (1.0 - velocity) + diff * velocity


def get_target_calculation_strategy(strategy_name: str):
    """名前からターゲット計算メソッドを取得する。"""
    strategies = {
        "subtraction": target_subtraction,
        "addition": target_addition,
        "multiplication": target_multiplication,
        "mix": target_mix,
    }

    # 拡張機能から登録された戦略を適用
    from module.extension_manager import get_extension_target_strategies

    ext_strats = get_extension_target_strategies()
    strategies.update(ext_strats)

    if strategy_name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"未知のターゲット計算方式: {strategy_name}。利用可能: {available}"
        )
    return strategies[strategy_name]


# --- Normalization Strategies ---


@merge_method
def normalize_std_mean(
    target_original: Parameter(Tensor),
    merged: Parameter(Tensor),
) -> Return(Tensor):
    """std/mean マッチング正規化。
    処理後のテンソル(merged)を元の統計量(target_original)に合わせて正規化する。
    """
    eps = 1e-7

    to = (
        target_original
        if torch.is_floating_point(target_original)
        else target_original.float()
    )
    mo = merged if torch.is_floating_point(merged) else merged.float()

    orig_std, orig_mean = torch.std_mean(to)
    new_std, new_mean = torch.std_mean(mo)

    # NaN 等除けのためのmax(..., eps)利用
    orig_std = torch.clamp(orig_std, min=eps)
    new_std = torch.clamp(new_std, min=eps)

    return (mo - (new_mean - orig_mean)) * (orig_std / new_std)


def get_normalization_calculation_strategy(strategy_name: str):
    """名前から正規化メソッドを取得する。パススルーの場合は None を返す。"""
    if strategy_name == "none":
        return None

    strategies = {
        "match_std_mean": normalize_std_mean,
    }
    if strategy_name not in strategies:
        available = "none, " + ", ".join(strategies.keys())
        raise ValueError(
            f"未知のノーマライズ方式: {strategy_name}。利用可能: {available}"
        )
    return strategies[strategy_name]
