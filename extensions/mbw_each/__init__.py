import logging
import json
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
from module.extension_manager import register_strategy, register_pre_config_hook

# --- MBW Each (個別重み) ストラテジー ---
# 1つの辞書 (JSON文字列) に各ブロックパターンの A用/B用の比率を含め、
# テンソル単位でパースして計算する戦略

_CACHE_MBW_EACH = {}


@merge_method
def strategy_mbw_each(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "{}",  # {"IN00": {"pattern": "...", "a": 0.5, "b": 0.3}, ...}
    **kwargs,
) -> Return(Tensor):
    """
    MBW Each 戦略。
    モデルA と モデルB に対して個別の係数 (alpha, beta) を適用して加算する。
    """
    key = kwargs.get("key", "")

    if key_patterns_json not in _CACHE_MBW_EACH:
        _CACHE_MBW_EACH[key_patterns_json] = json.loads(key_patterns_json)

    rules = _CACHE_MBW_EACH[key_patterns_json]

    alpha = 1.0
    beta = 1.0
    matched = False

    for rule_name, rule_data in rules.items():
        if rule_data["pattern"] in key:
            alpha = float(rule_data.get("a", 1.0))
            beta = float(rule_data.get("b", 1.0))
            matched = True
            break

    # マッチしなかったキー（指定外）はデフォルトで加算か、あるいはベースalphaを使うか。
    # 互換性のため A*(1-v) + B*v の通常のmix的な挙動をデフォルトとする。
    if not matched:
        v = float(velocity.item() if isinstance(velocity, Tensor) else velocity)
        return a * (1.0 - v) + b * v

    return a * alpha + b * beta


def mbw_each_pre_config_hook(config: dict) -> dict:
    """
    YAML設定から mbw_each をパースし、strategy: mbw_each を持つ model エントリに変換する。

    YAML 例:
    models:
      - left: "model_A.safetensors"
        right: "model_B.safetensors"
        strategy: "mbw_each"
        mbw_a: "1,0.5,0.5,..." # 26 or 20 要素
        mbw_b: "0,0.5,0.5,..." # 26 or 20 要素
    """
    from extensions.supermerger_mbw import _SD15_KEYWORDS, _SDXL_KEYWORDS

    models = config.get("models", [])
    for model_entry in models:
        mbw_a_str = model_entry.get("mbw_a", "")
        mbw_b_str = model_entry.get("mbw_b", "")

        # mbw_a / mbw_b の指定がある場合のみ、ブロックごとの重み(key_patterns)を生成する
        if mbw_a_str and mbw_b_str:
            try:
                ratios_a = [float(r.strip()) for r in mbw_a_str.split(",")]
                ratios_b = [float(r.strip()) for r in mbw_b_str.split(",")]
            except ValueError:
                logging.error("MBW Each パースエラー。数値とカンマのみを使用してください。")
                continue

            if len(ratios_a) != len(ratios_b):
                logging.error("mbw_a と mbw_b の要素数が一致しません。")
                continue

            if len(ratios_a) == 26:
                keywords = _SD15_KEYWORDS
            elif len(ratios_a) == 20:
                keywords = _SDXL_KEYWORDS
            else:
                logging.error(f"MBW Each: ブロック数が 26 または 20 ではありません (現在: {len(ratios_a)})")
                continue

            rules = {}
            for i, (ratio_a, ratio_b, pattern) in enumerate(zip(ratios_a, ratios_b, keywords)):
                rules[f"block_{i}"] = {"pattern": pattern, "a": ratio_a, "b": ratio_b}

            model_entry["key_patterns"] = rules  # mbw_each戦略側でJSONデコードして使用する

            # クリーンアップ
            model_entry.pop("mbw_a", None)
            model_entry.pop("mbw_b", None)

    return config


def setup():
    logging.info("MBW Each Extension の初期化を開始します。")
    register_pre_config_hook(mbw_each_pre_config_hook)
    register_strategy("mbw_each", strategy_mbw_each)
    logging.info("MBW Each Extension の登録が完了しました。")
