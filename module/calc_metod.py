import logging
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import torch


class CalculationStrategy(ABC):
    @abstractmethod
    def calculate(
        self,
        previous_model: Optional[Dict[str, torch.Tensor]],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        pass


class SubtractionStrategy(CalculationStrategy):
    def calculate(
        self,
        previous_model: Optional[Dict[str, torch.Tensor]],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = {}
        common_keys = set(left_model.keys()) & set(right_model.keys())

        for key in common_keys:
            if key_patterns is None or any(k in key for k in key_patterns):
                model_diff[key] = (left_model[key] - right_model[key]) * velocity

        return model_diff


class AdditionStrategy(CalculationStrategy):
    def calculate(
        self,
        previous_model: Optional[Dict[str, torch.Tensor]],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_sum = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    model_sum[key] = (left_model[key] + right_model[key]) * velocity
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_sum


class MultiplicationStrategy(CalculationStrategy):
    def calculate(
        self,
        previous_model: Optional[Dict[str, torch.Tensor]],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_prod = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    model_prod[key] = (left_model[key] * right_model[key]) * velocity
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_prod


class AverageStrategy(CalculationStrategy):
    def calculate(
        self,
        previous_model: Optional[Dict[str, torch.Tensor]],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_avg = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    model_avg[key] = (left_model[key] + right_model[key]) / 2 * velocity
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_avg


class MixStrategy(CalculationStrategy):
    def calculate(
        self,
        previous_model: Optional[Dict[str, torch.Tensor]],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_mix = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    model_mix[key] = (
                        left_model[key] * (1.0 - velocity) + right_model[key] * velocity
                    )
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_mix


class ReplaceStrategy(CalculationStrategy):
    def __init__(self, replace_with: str):
        self.replace_with = replace_with

    def calculate(
        self,
        previous_model: Optional[Dict[str, torch.Tensor]],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        if self.replace_with == "left":
            return {k: v * velocity for k, v in left_model.items()}
        elif self.replace_with == "right":
            return {k: v * velocity for k, v in right_model.items()}
        else:
            raise ValueError(f"未知の置き換えオプション: {self.replace_with}")


def get_calculation_strategy(
    strategy_name: str, replace_with: str = None
) -> CalculationStrategy:
    if strategy_name == "subtraction":
        return SubtractionStrategy()
    elif strategy_name == "addition":
        return AdditionStrategy()
    elif strategy_name == "multiplication":
        return MultiplicationStrategy()
    elif strategy_name == "average":
        return AverageStrategy()
    elif strategy_name == "replace":
        if replace_with is None:
            raise ValueError("置き換えオプションが指定されていません。")
        return ReplaceStrategy(replace_with)
    elif strategy_name == "mix":
        return MixStrategy()
    else:
        raise ValueError(f"未知の計算方式: {strategy_name}")
