"""SD モデルのターゲットモデルに対する差分適用ストラテジーモジュール。

left_model と right_model の差分を target_model に適用するロジック
（加算、減算、乗算、混合、角度ベース）と、結果の正規化ロジック
（パススルー、std/mean マッチング）を Strategy パターンで提供する。
"""

from abc import ABC, abstractmethod
import logging
from typing import Callable, Dict, Iterable, Optional

import torch

from module.calc_method import CalculationStrategy


class TargetCalculationStrategy(ABC):
    """target_model に差分を適用する計算ストラテジーの基底クラス。

    サブクラスで `_apply_diff` を実装し、各キーに対する演算を定義する。
    共通のイテレーション・進捗報告・キーチェックは基底クラスで行う。
    """

    def __init__(self) -> None:
        super().__init__()
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, progress_callback: Optional[Callable] = None) -> None:
        """進捗報告コールバックを設定する。"""
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
        left_model: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """target_model に left/right の差分を適用する。

        Args:
            target_model: 差分適用先のモデル state_dict。
            left_model: 左側モデルの state_dict。
            right_model: 右側モデルの state_dict。
            left_right_strategy: left/right 間の差分計算ストラテジー。
            left_right_velocity: left/right 間の計算強度。
            velocity: target への適用強度。
            target_layer_list: 処理対象レイヤー名パターンのリスト。

        Returns:
            差分適用後の target_model。
        """
        pass

    def _post_operation(self) -> None:
        """1キーの処理完了後に進捗コールバックを呼び出す。"""
        if self.progress_callback is not None:
            self.progress_callback()


class _DiffBasedTargetStrategy(TargetCalculationStrategy):
    """left/right の差分を計算し、target_model に適用する共通基底クラス。

    サブクラスでは `_apply_diff` のみを実装すればよい。
    """

    @abstractmethod
    def _apply_diff(
        self,
        target_value: torch.Tensor,
        diff_value: torch.Tensor,
        velocity: float,
    ) -> torch.Tensor:
        """target のテンソルに diff を適用する個別演算を定義する。

        Args:
            target_value: target_model の該当キーのテンソル。
            diff_value: left/right 間の差分テンソル。
            velocity: 適用強度。

        Returns:
            演算結果のテンソル。
        """
        pass

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
        model_diff = left_right_strategy.calculate(left_model, right_model, left_right_velocity, target_layer_list)
        for key in model_diff.keys():
            if key in target_model:
                if any(k in key for k in target_layer_list):
                    target_model[key] = self._apply_diff(target_model[key], model_diff[key], velocity)
            else:
                logging.warning(f"ターゲットモデルにキー {key} が見つかりません。スキップします。")
            self._post_operation()
        return target_model


class TargetAdditionStrategy(_DiffBasedTargetStrategy):
    """加算ストラテジー: target += diff * velocity"""

    def _apply_diff(self, target_value: torch.Tensor, diff_value: torch.Tensor, velocity: float) -> torch.Tensor:
        return target_value + diff_value * velocity


class TargetSubtractionStrategy(_DiffBasedTargetStrategy):
    """減算ストラテジー: target -= diff * velocity"""

    def _apply_diff(self, target_value: torch.Tensor, diff_value: torch.Tensor, velocity: float) -> torch.Tensor:
        return target_value - diff_value * velocity


class TargetMultiplicationStrategy(_DiffBasedTargetStrategy):
    """乗算ストラテジー: target *= diff * velocity"""

    def _apply_diff(self, target_value: torch.Tensor, diff_value: torch.Tensor, velocity: float) -> torch.Tensor:
        return target_value * diff_value * velocity


class TargetMixStrategy(_DiffBasedTargetStrategy):
    """混合ストラテジー: target * (1 - velocity) + diff * velocity"""

    def _apply_diff(self, target_value: torch.Tensor, diff_value: torch.Tensor, velocity: float) -> torch.Tensor:
        return target_value * (1.0 - velocity) + diff_value * velocity


class TargetAngleStrategy(TargetCalculationStrategy):
    """角度ベースのマージストラテジー。

    left と right のベクトル間の角度を使って、target を加重平均する。
    角度が小さい（方向が近い）部分ほど平均に近づける。
    """

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
        model_diff_l = left_right_strategy.calculate(left_model, target_model, left_right_velocity, target_layer_list)
        model_diff_r = left_right_strategy.calculate(right_model, target_model, left_right_velocity, target_layer_list)

        for key in model_diff_l.keys():
            if key in model_diff_r:
                if any(k in key for k in target_layer_list):
                    norm_prod = torch.norm(model_diff_l[key], dim=-1) * torch.norm(model_diff_r[key], dim=-1)
                    theta = ((model_diff_l[key] * model_diff_r[key]).sum(dim=-1) / norm_prod.clamp(min=1e-6)).unsqueeze(
                        -1
                    )
                    t = (2.0 * torch.cos(theta)) / (1.0 + torch.cos(theta))
                    avg = (left_model[key] + right_model[key]) * 0.5
                    target_model[key] = target_model[key] * (1.0 - t) + avg * t
                    del norm_prod, theta, t, avg
            else:
                logging.warning(f"ターゲットモデルにキー {key} が見つかりません。スキップします。")
            self._post_operation()

        del model_diff_l, model_diff_r
        return target_model


# --- Normalization Strategies ---


class TargetNormalizationCalculationStrategy(ABC):
    """ターゲット計算後の正規化ストラテジーの基底クラス。"""

    def __init__(self) -> None:
        super().__init__()
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, progress_callback: Optional[Callable] = None) -> None:
        """進捗報告コールバックを設定する。"""
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
        left_model: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """正規化付きでターゲット計算を実行する。

        Args:
            target_model: 差分適用先のモデル state_dict。
            left_model: 左側モデルの state_dict。
            right_model: 右側モデルの state_dict。
            target_strategy: ターゲット計算ストラテジー。
            left_right_strategy: left/right 間の差分計算ストラテジー。
            left_right_velocity: left/right 間の計算強度。
            velocity: target への適用強度。
            target_layer_list: 処理対象レイヤー名パターンのリスト。

        Returns:
            正規化後の target_model。
        """
        pass


class TargetNormalizationPassthrough(TargetNormalizationCalculationStrategy):
    """正規化なし（パススルー）。ターゲット計算結果をそのまま返す。"""

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
        left_model: Optional[Dict[str, torch.Tensor]] = None,
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
            left_model,
        )


class TargetNormalizationMatchStdMean(TargetNormalizationCalculationStrategy):
    """std/mean マッチング正規化。

    ターゲット計算の前後で各テンソルの標準偏差と平均を保存し、
    処理後のテンソルを元の統計量に合わせて正規化する。
    """

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
        left_model: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # 対象キーのみの統計量を事前計算
        target_keys = {k for k in target_model.keys() if any(pat in k for pat in key_patterns)}

        orig_std_mean = {}
        for k in target_keys:
            v = target_model[k]
            if not torch.is_floating_point(v):
                v = v.float()
            orig_std_mean[k] = torch.std_mean(v)

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
            left_model,
        )

        # 対象キーのみ正規化を適用
        for key in target_keys:
            if key not in processed:
                continue
            v = processed[key]
            if not torch.is_floating_point(v):
                v = v.float()
                processed[key] = v

            new_std, new_mean = torch.std_mean(v)
            orig_std, orig_mean = orig_std_mean[key]

            processed[key] = (v - (new_mean - orig_mean)) * (max(orig_std, self.eps) / max(new_std, self.eps))

        return processed


# --- Factory Functions ---


_TARGET_STRATEGY_MAP: Dict[str, Callable[[], TargetCalculationStrategy]] = {
    "subtraction": TargetSubtractionStrategy,
    "addition": TargetAdditionStrategy,
    "multiplication": TargetMultiplicationStrategy,
    "mix": TargetMixStrategy,
    "angle": TargetAngleStrategy,
}

_NORMALIZATION_STRATEGY_MAP: Dict[str, Callable[[], TargetNormalizationCalculationStrategy]] = {
    "none": TargetNormalizationPassthrough,
    "match_std_mean": TargetNormalizationMatchStdMean,
}


def get_target_calculation_strategy(strategy_name: str) -> TargetCalculationStrategy:
    """名前からターゲット計算ストラテジーのインスタンスを取得する。

    Args:
        strategy_name: ストラテジー名
            ('subtraction', 'addition', 'multiplication', 'mix', 'angle')。

    Returns:
        対応する TargetCalculationStrategy のインスタンス。

    Raises:
        ValueError: 未知のストラテジー名が指定された場合。
    """
    factory = _TARGET_STRATEGY_MAP.get(strategy_name)
    if factory is None:
        available = ", ".join(_TARGET_STRATEGY_MAP.keys())
        raise ValueError(f"未知のターゲット計算方式: {strategy_name}。利用可能: {available}")
    return factory()


def get_normalization_calculation_strategy(
    strategy_name: str,
) -> TargetNormalizationCalculationStrategy:
    """名前から正規化ストラテジーのインスタンスを取得する。

    Args:
        strategy_name: ストラテジー名 ('none', 'match_std_mean')。

    Returns:
        対応する TargetNormalizationCalculationStrategy のインスタンス。

    Raises:
        ValueError: 未知のストラテジー名が指定された場合。
    """
    factory = _NORMALIZATION_STRATEGY_MAP.get(strategy_name)
    if factory is None:
        available = ", ".join(_NORMALIZATION_STRATEGY_MAP.keys())
        raise ValueError(f"未知のノーマライズ方式: {strategy_name}。利用可能: {available}")
    return factory()
