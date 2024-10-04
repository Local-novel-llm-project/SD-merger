import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Optional

import torch

from module.calc_metod import CalculationStrategy


class TargetCalculationStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.progress_callback = None

    def set_progress_callback(
        self, progress_callback: Optional[Callable] = None
    ) -> None:
        self.progress_callback = progress_callback

    @abstractmethod
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        pass

    def post_operation(self):
        if self.progress_callback is not None:
            self.progress_callback()


class TargetNormalizationCalculationStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.progress_callback = None

    def set_progress_callback(
        self, progress_callback: Optional[Callable] = None
    ) -> None:
        self.progress_callback = progress_callback

    @abstractmethod
    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        target_strategy: TargetCalculationStrategy,
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        pass


class TargetNormalizationPassthrough(TargetNormalizationCalculationStrategy):
    def __init__(self):
        super().__init__()

    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        target_strategy: TargetCalculationStrategy,
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        if self.progress_callback is not None:
            target_strategy.set_progress_callback(self.progress_callback)
        return target_strategy.calculate(
            target_model,
            merged_model,
            right_model,
            left_right_strategy,
            left_right_velocity,
            velocity,
            key_patterns,
        )


class TargetNormalizationMatchStdMean(TargetNormalizationCalculationStrategy):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def calculate(
        self,
        target_model: Dict[str, torch.Tensor],
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        target_strategy: TargetCalculationStrategy,
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
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
            merged_model,
            right_model,
            left_right_strategy,
            left_right_velocity,
            velocity,
            key_patterns,
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
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        for key in merged_model.keys():
            if key in target_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    target_model[key] = target_model[key] + merged_model[key] * velocity
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
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        for key in merged_model.keys():
            if key in target_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    target_model[key] = target_model[key] - merged_model[key] * velocity
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
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        for key in merged_model.keys():
            if key in target_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    target_model[key] = target_model[key] * merged_model[key] * velocity
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
        merged_model: Dict[str, torch.Tensor],
        right_model: Optional[Dict[str, torch.Tensor]],
        left_right_strategy: CalculationStrategy,
        left_right_velocity: float,
        velocity: float,
        key_patterns: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        for key in merged_model.keys():
            if key in target_model.keys():
                if key_patterns is None or any(k in key for k in key_patterns):
                    target_model[key] = (
                        target_model[key] * (1.0 - velocity)
                        + merged_model[key] * velocity
                    )
            else:
                logging.warning(
                    f"ターゲットモデルにキー {key} が見つかりません。スキップします。"
                )
            self.post_operation()
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
