from abc import ABC, abstractmethod
import logging
from typing import Callable, Dict, Iterable

import torch

from module.calc_metod import CalculationStrategy


class TargetCalculationStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.progress_callback = None

    def set_progress_callback(self, progress_callback: Callable | None = None) -> None:
        self.progress_callback = progress_callback

    @abstractmethod
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        pass

    def post_operation(self):
        if self.progress_callback is not None:
            self.progress_callback()


class TargetNormalizationCalculationStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.progress_callback = None

    def set_progress_callback(self, progress_callback: Callable | None = None) -> None:
        self.progress_callback = progress_callback

    @abstractmethod
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        target_strategy: TargetCalculationStrategy,
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        pass


class TargetNormalizationPassthrough(TargetNormalizationCalculationStrategy):
    def __init__(self):
        super().__init__()

    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        target_strategy: TargetCalculationStrategy,
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        if self.progress_callback is not None:
            target_strategy.set_progress_callback(self.progress_callback)
        return target_strategy.calculate(
            target_model,
            left_model,
            right_model,
            left_right_strategy,
            left_right_velocity,
            velocity,
            target_layer_list,
        )


class TargetNormalizationMatchStdMean(TargetNormalizationCalculationStrategy):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        target_strategy: TargetCalculationStrategy,
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        # テンソルが浮動小数点数であることを確認
        for k, v in target_model.items():
            if not torch.is_floating_point(v):
                target_model[k] = v.float()

        orig_std_mean = {k: torch.std_mean(v) for k, v in target_model.items()}

        if self.progress_callback is not None:
            target_strategy.set_progress_callback(self.progress_callback)
        processed = target_strategy.calculate(
            target_model,
            left_model,
            right_model,
            left_right_strategy,
            left_right_velocity,
            velocity,
            target_layer_list,
        )

        # テンソルが浮動小数点数であることを確認
        for k, v in processed.items():
            if not torch.is_floating_point(v):
                processed[k] = v.float()

        std_mean = {k: torch.std_mean(v) for k, v in processed.items()}

        for key in processed:
            if key in orig_std_mean:
                processed[key] = (
                    processed[key] - (std_mean[key][1] - orig_std_mean[key][1])
                ) * (
                    max(std_mean[key][0], self.eps)
                    / max(orig_std_mean[key][0], self.eps)
                )
        return processed


class TargetAdditionStrategy(TargetCalculationStrategy):
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = left_right_strategy.calculate(
            left_model, right_model, left_right_velocity, target_layer_list
        )
        for key in model_diff.keys():
            if key in target_model.keys():
                if any(k in key for k in target_layer_list):
                    # model_diff[key] = (left_model[key] + right_model[key])*velocity
                    target_model[key] = target_model[key] + model_diff[key] * velocity
            else:
                logging.warning(
                    f"ターゲットモデルにキー {key} が見つかりません。スキップします。"
                )
            self.post_operation()
        return target_model


class TargetSubtractionStrategy(TargetCalculationStrategy):
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = left_right_strategy.calculate(
            left_model, right_model, left_right_velocity, target_layer_list
        )
        for key in model_diff.keys():
            if key in target_model.keys():
                if any(k in key for k in target_layer_list):
                    target_model[key] = target_model[key] - model_diff[key] * velocity
                    # print(key)
            else:
                logging.warning(
                    f"ターゲットモデルにキー {key} が見つかりません。スキップします。"
                )
            self.post_operation()
        return target_model


class TargetMultiplicationStrategy(TargetCalculationStrategy):
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = left_right_strategy.calculate(
            left_model, right_model, left_right_velocity, target_layer_list
        )
        for key in model_diff.keys():
            if key in target_model.keys():
                if any(k in key for k in target_layer_list):
                    target_model[key] = target_model[key] * model_diff[key] * velocity
            else:
                logging.warning(
                    f"ターゲットモデルにキー {key} が見つかりません。スキップします。"
                )
            self.post_operation()
        return target_model


class TargetMixStrategy(TargetCalculationStrategy):
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        model_diff = left_right_strategy.calculate(
            left_model, right_model, left_right_velocity, target_layer_list
        )
        for key in model_diff.keys():
            if key in target_model.keys():
                if any(k in key for k in target_layer_list):
                    target_model[key] = (
                        target_model[key] * (1.0 - velocity)
                        + model_diff[key] * velocity
                    )
            else:
                logging.warning(
                    f"ターゲットモデルにキー {key} が見つかりません。スキップします。"
                )
            self.post_operation()
        return target_model


class TargetAngleStrategy(TargetCalculationStrategy):
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        target_layer_list: Iterable[str],
    ) -> Dict[str, torch.Tensor]:

        model_diff_l = left_right_strategy.calculate(
            left_model, target_model, left_right_velocity, target_layer_list
        )
        model_diff_r = left_right_strategy.calculate(
            right_model, target_model, left_right_velocity, target_layer_list
        )
        for key in model_diff_l.keys():
            if key in model_diff_r.keys():
                if any(k in key for k in target_layer_list):
                    norm_prod = torch.norm(model_diff_l[key], dim=-1) * torch.norm(
                        model_diff_r[key], dim=-1
                    )
                    theta = (
                        (model_diff_l[key] * model_diff_r[key]).sum(dim=-1)
                        / norm_prod.clamp(min=1e-6)
                    ).unsqueeze(-1)
                    t = (2.0 * torch.cos(theta)) / (1.0 + torch.cos(theta))
                    avg = (left_model[key] + right_model[key]) * 0.5
                    target_model[key] = target_model[key] * (1.0 - t) + avg * t
                    del norm_prod, theta, t, avg
            else:
                logging.warning(
                    f"ターゲットモデルにキー {key} が見つかりません。スキップします。"
                )
            self.post_operation()
        del model_diff_l, model_diff_r
        return target_model


def get_target_calculation_strategy(strategy_name: str) -> TargetCalculationStrategy:
    if strategy_name == "subtraction":
        return TargetSubtractionStrategy()
    elif strategy_name == "addition":
        return TargetAdditionStrategy()
    elif strategy_name == "multiplication":
        return TargetMultiplicationStrategy()
    elif strategy_name == "mix":
        return TargetMixStrategy()
    elif strategy_name == "angle":
        return TargetAngleStrategy()
    else:
        raise ValueError(f"未知のターゲット計算方式: {strategy_name}")


def get_normalization_calculation_strategy(
    strategy_name: str,
) -> TargetNormalizationCalculationStrategy:
    if strategy_name == "none":
        return TargetNormalizationPassthrough()
    elif strategy_name == "match_std_mean":
        return TargetNormalizationMatchStdMean()
    else:
        raise ValueError(f"未知のノーマライズ方式: {strategy_name}")
