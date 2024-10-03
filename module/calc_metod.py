import logging
from typing import Dict, Iterable
from abc import ABC, abstractmethod
import torch


class CalculationStrategy(ABC):
    @abstractmethod
    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocities: Dict[str, float],
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        pass


class SubtractionStrategy(CalculationStrategy):
    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocities: Dict[str, float],
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = {}
        common_keys = set(left_model.keys()) & set(right_model.keys())

        for key in common_keys:
            if any(k in key for k in target_layer_list):
                velocity = next(
                    (v for k, v in velocities.items() if k in key),
                    velocities.get("default", 1.0),
                )
                model_diff[key] = (left_model[key] - right_model[key]) * velocity

        return model_diff


class AdditionStrategy(CalculationStrategy):
    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocities: Dict[str, float],
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if any(k in key for k in target_layer_list):
                    velocity = next(
                        (v for k, v in velocities.items() if k in key),
                        velocities.get("default", 1.0),
                    )
                    model_diff[key] = (left_model[key] + right_model[key]) * velocity
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_diff


class MultiplicationStrategy(CalculationStrategy):
    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocities: Dict[str, float],
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if any(k in key for k in target_layer_list):
                    velocity = next(
                        (v for k, v in velocities.items() if k in key),
                        velocities.get("default", 1.0),
                    )
                    model_diff[key] = (left_model[key] * right_model[key]) * velocity
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_diff


class AverageStrategy(CalculationStrategy):
    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocities: Dict[str, float],
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if any(k in key for k in target_layer_list):
                    velocity = next(
                        (v for k, v in velocities.items() if k in key),
                        velocities.get("default", 1.0),
                    )
                    model_diff[key] = (
                        (left_model[key] + right_model[key]) / 2 * velocity
                    )
                # model_diff[key] = (left_model[key] + right_model[key]) / 2 * velocity
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_diff


class MixStrategy(CalculationStrategy):
    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocities: Dict[str, float],
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = {}
        for key in left_model.keys():
            if key in right_model.keys():
                if any(k in key for k in target_layer_list):
                    velocity = next(
                        (v for k, v in velocities.items() if k in key),
                        velocities.get("default", 1.0),
                    )
                    model_diff[key] = left_model[key] * (1.0 - velocity) + right_model[
                        key
                    ] * (velocity)
            else:
                logging.warning(
                    f"右モデルにキー {key} が見つかりません。スキップします。"
                )
        return model_diff


class ReplaceStrategy(CalculationStrategy):
    def __init__(self, replace_with: str):
        self.replace_with = replace_with

    def calculate(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        velocities: Dict[str, float],
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_to_use = left_model if self.replace_with == "left" else right_model
        result = {}
        for key, tensor in model_to_use.items():
            if any(k in key for k in target_layer_list):
                velocity = next(
                    (v for k, v in velocities.items() if k in key),
                    velocities.get("default", 1.0),
                )
                result[key] = tensor * velocity
        return result


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
