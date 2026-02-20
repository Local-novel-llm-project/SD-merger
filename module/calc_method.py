"""SD モデルの left/right 間の差分計算ストラテジーモジュール。

left_model と right_model の2つのモデル間で行う演算（減算、加算、乗算、
平均、混合、置き換え）を Strategy パターンで抽象化する。
"""

import logging
from typing import Callable, Dict, Iterable

import torch

from abc import ABC, abstractmethod


class CalculationStrategy(ABC):
    """left_model と right_model に対する差分計算の基底クラス。

    サブクラスで `_operation` を実装することで、各種計算ロジックを定義する。
    `calculate` メソッドは共通的なキーのフィルタリングとイテレーションを行う。
    """

    @abstractmethod
    def _operation(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        velocity: float,
    ) -> torch.Tensor:
        """left と right のテンソルに対して行う個別演算を定義する。

        Args:
            left: left_model のテンソル。
            right: right_model のテンソル。
            velocity: 計算の強度パラメータ。

        Returns:
            演算結果のテンソル。
        """
        pass

    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        """left_model と right_model の共通キーに対して差分計算を行う。

        Args:
            left_model: 左側モデルの state_dict。
            right_model: 右側モデルの state_dict。
            velocity: 計算の強度パラメータ。
            target_layer_list: 処理対象とするレイヤー名パターンのリスト。

        Returns:
            差分計算結果を格納した辞書。
        """
        if not target_layer_list:
            logging.warning("target_layer_list が空です。結果は空辞書になります。")
            return {}

        model_diff = {}
        common_keys = set(left_model.keys()) & set(right_model.keys())

        for key in common_keys:
            if any(k in key for k in target_layer_list):
                model_diff[key] = self._operation(left_model[key], right_model[key], velocity)

        return model_diff


class SubtractionStrategy(CalculationStrategy):
    """減算ストラテジー: (left - right) * velocity"""

    def _operation(self, left: torch.Tensor, right: torch.Tensor, velocity: float) -> torch.Tensor:
        return (left - right) * velocity


class AdditionStrategy(CalculationStrategy):
    """加算ストラテジー: (left + right) * velocity"""

    def _operation(self, left: torch.Tensor, right: torch.Tensor, velocity: float) -> torch.Tensor:
        return (left + right) * velocity


class MultiplicationStrategy(CalculationStrategy):
    """乗算ストラテジー: (left * right) * velocity"""

    def _operation(self, left: torch.Tensor, right: torch.Tensor, velocity: float) -> torch.Tensor:
        return (left * right) * velocity


class AverageStrategy(CalculationStrategy):
    """平均ストラテジー: (left + right) / 2 * velocity"""

    def _operation(self, left: torch.Tensor, right: torch.Tensor, velocity: float) -> torch.Tensor:
        return (left + right) / 2 * velocity


class MixStrategy(CalculationStrategy):
    """混合ストラテジー: left * (1 - velocity) + right * velocity"""

    def _operation(self, left: torch.Tensor, right: torch.Tensor, velocity: float) -> torch.Tensor:
        return left * (1.0 - velocity) + right * velocity


class ReplaceStrategy(CalculationStrategy):
    """置き換えストラテジー: 指定されたソース ('left' or 'right') のテンソルを使用する。

    Args:
        replace_with: 置き換え元を指定する文字列 ('left' or 'right')。
    """

    def __init__(self, replace_with: str):
        if replace_with not in ("left", "right"):
            raise ValueError(f"未知の置き換えオプション: {replace_with}。'left' または 'right' を指定してください。")
        self.replace_with = replace_with

    def _operation(self, left: torch.Tensor, right: torch.Tensor, velocity: float) -> torch.Tensor:
        source = left if self.replace_with == "left" else right
        return source * velocity

    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        """置き換え元のモデルのキーに対して処理を行う。

        通常の Strategy と異なり、共通キーではなくソース側のキーを基準にする。
        """
        if not target_layer_list:
            logging.warning("target_layer_list が空です。結果は空辞書になります。")
            return {}

        source = left_model if self.replace_with == "left" else right_model
        return {key: source[key] * velocity for key in source.keys() if any(k in key for k in target_layer_list)}


_STRATEGY_MAP: Dict[str, Callable[..., CalculationStrategy]] = {
    "subtraction": lambda **_: SubtractionStrategy(),
    "addition": lambda **_: AdditionStrategy(),
    "multiplication": lambda **_: MultiplicationStrategy(),
    "average": lambda **_: AverageStrategy(),
    "mix": lambda **_: MixStrategy(),
    "replace": lambda replace_with=None, **_: ReplaceStrategy(replace_with),
}


def get_calculation_strategy(strategy_name: str, replace_with: str = None) -> CalculationStrategy:
    """名前から計算ストラテジーのインスタンスを取得する。

    Args:
        strategy_name: ストラテジー名
            ('subtraction', 'addition', 'multiplication', 'average', 'mix', 'replace')。
        replace_with: 'replace' ストラテジー用の置き換え元 ('left' or 'right')。

    Returns:
        対応する CalculationStrategy のインスタンス。

    Raises:
        ValueError: 未知のストラテジー名が指定された場合。
    """
    factory = _STRATEGY_MAP.get(strategy_name)
    if factory is None:
        available = ", ".join(_STRATEGY_MAP.keys())
        raise ValueError(f"未知の計算方式: {strategy_name}。利用可能: {available}")

    if strategy_name == "replace" and replace_with is None:
        raise ValueError("'replace' ストラテジーには replace_with の指定が必須です。")

    return factory(replace_with=replace_with)
