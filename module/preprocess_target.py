import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable

import torch
from safetensors import safe_open

from lib.hungarian_algorithm import (
    hungarian_algorithm_low_mem,
)
from module.utility import load_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SIZE_THRESHOLD = 2000


class TargetPreprocessingStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.progress_callback = None

    def set_progress_callback(
        self, progress_callback: Callable[[], None] = None
    ) -> None:
        self.progress_callback = progress_callback

    @abstractmethod
    def preprocess(
        self,
        target_model: Dict[str, torch.Tensor],
        target_layer_list: Iterable[str],
        reference_model: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pass

    def post_operation(self):
        if self.progress_callback is not None:
            self.progress_callback()


class NoOpTargetPreprocessingStrategy(TargetPreprocessingStrategy):
    """
    何もしないターゲット前処理戦略。デフォルトの動作として使用します。
    """

    def preprocess(
        self,
        target_model: Dict[str, torch.Tensor],
        target_layer_list: Iterable[str],
        reference_model: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return target_model


class WeightMatchingTargetStrategy(TargetPreprocessingStrategy):
    """
    ターゲットモデルと参照モデル間で重みのマッチングを行う前処理戦略。
    論文「Git Re-Basin」の手法に基づき、ゼロ除算の問題を解決し、データ型の不一致を修正します。
    """

    def __init__(self, max_iter: int = 10, cache_dir: str = "./cache_target"):
        super().__init__()
        self.max_iter = max_iter
        self.cache_dir = cache_dir

    def preprocess(
        self,
        target_model_path: str,
        target_layer_list: Iterable[str],
        reference_model_path: str,
    ) -> str:
        logger.debug("WeightMatchingTargetStrategy の preprocess を開始します。")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"使用デバイス: {device}")

        # キャッシュディレクトリを作成
        os.makedirs(self.cache_dir, exist_ok=True)

        # ターゲットモデルのキーを取得
        with safe_open(target_model_path, framework="pt", device="cpu") as target_f:
            target_keys = target_f.keys()

        # 参照モデルのキーを取得
        with safe_open(reference_model_path, framework="pt", device="cpu") as ref_f:
            reference_keys = ref_f.keys()

        # 重みとバイアスのキーを取得
        weight_keys = [
            k
            for k in target_keys
            if any(layer in k for layer in target_layer_list) and "weight" in k
        ]

        for weight_key in weight_keys:
            logger.debug(f"処理対象のキー: {weight_key}")
            if weight_key in reference_keys:
                # テンソルを読み込む
                target_weight = load_tensor(target_model_path, weight_key).to(
                    device, dtype=torch.float16
                )
                reference_weight = load_tensor(reference_model_path, weight_key).to(
                    device, dtype=torch.float16
                )
                logger.debug(
                    f"target_weight の形状: {target_weight.shape}, dtype: {target_weight.dtype}"
                )
                logger.debug(
                    f"reference_weight の形状: {reference_weight.shape}, dtype: {reference_weight.dtype}"
                )

                # 対応するバイアスキー
                bias_key = weight_key.replace("weight", "bias")
                if bias_key in target_keys and bias_key in reference_keys:
                    target_bias = load_tensor(target_model_path, bias_key).to(
                        device, dtype=torch.float16
                    )
                    reference_bias = load_tensor(reference_model_path, bias_key).to(
                        device, dtype=torch.float16
                    )
                    logger.debug(f"バイアス {bias_key} を処理します。")
                else:
                    target_bias = None
                    reference_bias = None
                    logger.debug(
                        f"バイアス {bias_key} が見つからないためスキップします。"
                    )

                # ユニット数
                num_units = target_weight.shape[0]
                logger.debug(f"ユニット数: {num_units}")

                # 初期Permutationはインデックスの配列
                P_indices = torch.arange(num_units, device=device, dtype=torch.long)

                for iter_num in range(self.max_iter):
                    logger.debug(f"{iter_num+1} 回目の反復を開始します。")

                    # 重みとバイアスのPermutation適用
                    permuted_weight = target_weight[P_indices]
                    if target_bias is not None:
                        permuted_bias = target_bias[P_indices]
                    else:
                        permuted_bias = None

                    # 重みを2次元に変形
                    permuted_weight_flat = permuted_weight.view(
                        permuted_weight.shape[0], -1
                    )
                    reference_weight_flat = reference_weight.view(
                        reference_weight.shape[0], -1
                    )

                    # コサイン類似度の計算
                    # 正規化（eps を指定してゼロ除算を防止）
                    permuted_weight_norm = torch.nn.functional.normalize(
                        permuted_weight_flat, dim=1, eps=1e-7
                    )
                    reference_weight_norm = torch.nn.functional.normalize(
                        reference_weight_flat, dim=1, eps=1e-7
                    )

                    # 類似度行列計算
                    similarity = torch.mm(
                        permuted_weight_norm, reference_weight_norm.t()
                    )

                    # 無効な値が含まれていないかチェック
                    if torch.isnan(similarity).any() or torch.isinf(similarity).any():
                        logger.error(
                            "類似度行列に無効な値 (NaN または Inf) が含まれています。"
                        )
                        break

                    # コスト行列を生成（最大化問題を最小化問題に変換）
                    cost_matrix = -similarity.detach()
                    matrix_size = cost_matrix.size(0)

                    ans_pos = hungarian_algorithm_low_mem(cost_matrix)
                    col_ind = torch.tensor([pos[1] for pos in ans_pos], device=device)
                    new_P_indices = col_ind
                    logger.debug("Sinkhorn-Knopp アルゴリズムを使用しました。")
                    # 収束判定
                    if torch.equal(P_indices, new_P_indices):
                        logger.debug("Permutation が収束しました。")
                        break

                    P_indices = new_P_indices
                    logger.debug("Permutation を更新しました。")

                # 最終的なPermutationを適用
                matched_target_weight = target_weight[P_indices]
                # 処理結果をキャッシュに保存
                cache_path = os.path.join(self.cache_dir, f"{weight_key}.pt")
                torch.save(matched_target_weight, cache_path)
                logger.debug(
                    f"重みをPermutationし、キャッシュに保存しました: {cache_path}"
                )

                # **GPUメモリからテンソルを削除**
                del target_weight, reference_weight, matched_target_weight
                torch.cuda.empty_cache()

                if target_bias is not None:
                    matched_target_bias = target_bias[P_indices].cpu()
                    cache_bias_path = os.path.join(self.cache_dir, f"{bias_key}.pt")
                    torch.save(matched_target_bias, cache_bias_path)
                    logger.debug(
                        f"バイアスをPermutationし、キャッシュに保存しました: {cache_bias_path}"
                    )

                    # **GPUメモリからテンソルを削除**
                    del target_bias, reference_bias, matched_target_bias
                    torch.cuda.empty_cache()

                self.post_operation()
            else:
                logger.warning(
                    f"参照モデルにキー {weight_key} が見つからないため、ターゲットモデルの重みを使用します。"
                )
                # ターゲットモデルの重みをキャッシュに保存
                target_weight = load_tensor(target_model_path, weight_key).cpu()
                cache_path = os.path.join(self.cache_dir, f"{weight_key}.pt")
                torch.save(target_weight, cache_path)
                logger.debug(
                    f"ターゲットモデルの重みをキャッシュに保存しました: {cache_path}"
                )
                del target_weight
                torch.cuda.empty_cache()

                # 同様にバイアスも処理
                bias_key = weight_key.replace("weight", "bias")
                if bias_key in target_keys:
                    target_bias = load_tensor(target_model_path, bias_key).cpu()
                    cache_bias_path = os.path.join(self.cache_dir, f"{bias_key}.pt")
                    torch.save(target_bias, cache_bias_path)
                    logger.debug(
                        f"ターゲットモデルのバイアスをキャッシュに保存しました: {cache_bias_path}"
                    )
                    del target_bias
                    torch.cuda.empty_cache()

        logger.debug("WeightMatchingTargetStrategy の preprocess が完了しました。")
        return self.cache_dir  # キャッシュディレクトリのパスを返す


def get_target_preprocessing_strategy(
    strategy_name: str,
) -> TargetPreprocessingStrategy:
    if strategy_name == "noop":
        return NoOpTargetPreprocessingStrategy()
    elif strategy_name == "weight_matching":
        return WeightMatchingTargetStrategy()
    else:
        raise ValueError(f"未知のターゲット前処理戦略: {strategy_name}")
