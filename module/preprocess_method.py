import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Tuple

import torch
from safetensors import safe_open

from lib.hungarian_algorithm import hungarian_algorithm_low_mem
from module.utility import load_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SIZE_THRESHOLD = 2000


class PreprocessingStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.progress_callback = None

    def set_progress_callback(
        self, progress_callback: Callable[[int], None] = None
    ) -> None:
        self.progress_callback = progress_callback

    @abstractmethod
    def preprocess(
        self,
        left_model_path: str,
        right_model_path: str,
        target_layer_list: Iterable[str],
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        pass

    def post_operation(self):
        if self.progress_callback is not None:
            self.progress_callback()


class NoOpPreprocessingStrategy(PreprocessingStrategy):
    """
    何もしない前処理戦略。デフォルトの動作として使用します。
    """

    def preprocess(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        target_layer_list: Iterable[str],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return left_model, right_model


class WeightMatchingStrategy(PreprocessingStrategy):
    """
    重みのマッチングを行う前処理戦略。
    左右のモデル間でユニットの再配置を行い、重みを調整します。
    """

    def __init__(self, max_iter: int = 10, cache_dir: str = "./cache"):
        super().__init__()
        self.max_iter = max_iter
        self.cache_dir = cache_dir

    def preprocess(
        self,
        left_model_path: str,
        right_model_path: str,
        target_layer_list: Iterable[str],
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        logger.debug("WeightMatchingStrategy の preprocess を開始します。")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"使用デバイス: {device}")

        # キャッシュディレクトリを作成
        os.makedirs(self.cache_dir, exist_ok=True)

        # 左モデルのキーを取得
        with safe_open(left_model_path, framework="pt", device="cpu") as left_f:
            left_keys = left_f.keys()

        # 右モデルのキーを取得
        with safe_open(right_model_path, framework="pt", device="cpu") as right_f:
            right_keys = right_f.keys()

        # 重みとバイアスのキーを取得
        weight_keys = [
            k
            for k in left_keys
            if any(layer in k for layer in target_layer_list) and "weight" in k
        ]

        for weight_key in weight_keys:
            logger.debug(f"処理対象のキー: {weight_key}")
            if weight_key in right_keys:
                # テンソルを読み込む
                left_weight = load_tensor(left_model_path, weight_key).to(
                    device, dtype=torch.float16
                )
                right_weight = load_tensor(right_model_path, weight_key).to(
                    device, dtype=torch.float16
                )
                logger.debug(
                    f"left_weight の形状: {left_weight.shape}, dtype: {left_weight.dtype}"
                )
                logger.debug(
                    f"right_weight の形状: {right_weight.shape}, dtype: {right_weight.dtype}"
                )

                # 対応するバイアスキー
                bias_key = weight_key.replace("weight", "bias")
                if bias_key in left_keys and bias_key in right_keys:
                    left_bias = load_tensor(left_model_path, bias_key).to(
                        device, dtype=torch.float16
                    )
                    right_bias = load_tensor(right_model_path, bias_key).to(
                        device, dtype=torch.float16
                    )
                    logger.debug(f"バイアス {bias_key} を処理します。")
                else:
                    left_bias = None
                    right_bias = None
                    logger.debug(
                        f"バイアス {bias_key} が見つからないためスキップします。"
                    )

                # ユニット数
                num_units = left_weight.shape[0]
                logger.debug(f"ユニット数: {num_units}")

                # 初期Permutationはインデックスの配列
                P_indices = torch.arange(num_units, device=device, dtype=torch.long)

                for iter_num in range(self.max_iter):
                    logger.debug(f"{iter_num+1} 回目の反復を開始します。")

                    # 重みとバイアスのPermutation適用
                    permuted_right_weight = right_weight[P_indices]
                    if right_bias is not None:
                        permuted_right_bias = right_bias[P_indices]
                    else:
                        permuted_right_bias = None

                    # 重みを2次元に変形
                    left_weight_flat = left_weight.view(left_weight.shape[0], -1)
                    permuted_right_weight_flat = permuted_right_weight.view(
                        permuted_right_weight.shape[0], -1
                    )

                    # コサイン類似度の計算
                    # 正規化（eps を指定してゼロ除算を防止）
                    left_weight_norm = torch.nn.functional.normalize(
                        left_weight_flat, dim=1, eps=1e-7
                    )
                    logger.debug(f"left_weight_norm の形状: {left_weight_norm.shape}")
                    right_weight_norm = torch.nn.functional.normalize(
                        permuted_right_weight_flat, dim=1, eps=1e-7
                    )
                    logger.debug(
                        f"permuted_right_weight_norm の形状: {right_weight_norm.shape}"
                    )
                    # 類似度行列計算
                    similarity = torch.mm(left_weight_norm, right_weight_norm.t())
                    logger.debug(f"similarity の形状: {similarity.shape}")
                    # 無効な値が含まれていないかチェック
                    if torch.isnan(similarity).any() or torch.isinf(similarity).any():
                        logger.error(
                            "類似度行列に無効な値 (NaN または Inf) が含まれています。"
                        )
                        break

                    # コスト行列を生成（最大化問題を最小化問題に変換）
                    cost_matrix = -similarity.detach()
                    ans_pos = hungarian_algorithm_low_mem(cost_matrix)
                    col_ind = torch.tensor([pos[1] for pos in ans_pos], device=device)
                    new_P_indices = col_ind
                    # 収束判定
                    if torch.equal(P_indices, new_P_indices):
                        logger.debug("Permutation が収束しました。")
                        break

                    P_indices = new_P_indices
                    logger.debug("Permutation を更新しました。")

                # 最終的なPermutationを適用
                matched_right_weight = right_weight[P_indices]
                # 処理結果をキャッシュに保存
                cache_path = os.path.join(self.cache_dir, f"{weight_key}.pt")
                torch.save(matched_right_weight, cache_path)
                logger.debug(
                    f"重みをPermutationし、キャッシュに保存しました: {cache_path}"
                )

                # **GPUメモリからテンソルを削除**
                del left_weight, right_weight, matched_right_weight
                torch.cuda.empty_cache()

                if right_bias is not None:
                    matched_right_bias = right_bias[P_indices]
                    cache_bias_path = os.path.join(self.cache_dir, f"{bias_key}.pt")
                    torch.save(matched_right_bias, cache_bias_path)
                    logger.debug(
                        f"バイアスをPermutationし、キャッシュに保存しました: {cache_bias_path}"
                    )

                    # **GPUメモリからテンソルを削除**
                    del left_bias, right_bias, matched_right_bias
                    torch.cuda.empty_cache()

                self.post_operation()
            else:
                logger.warning(
                    f"右モデルにキー {weight_key} が見つからないため、左モデルの重みを使用します。"
                )
                # 左モデルの重みをキャッシュに保存
                left_weight = load_tensor(left_model_path, weight_key)
                cache_path = os.path.join(self.cache_dir, f"{weight_key}.pt")
                torch.save(left_weight, cache_path)
                logger.debug(f"左モデルの重みをキャッシュに保存しました: {cache_path}")
                del left_weight
                torch.cuda.empty_cache()

                # 同様にバイアスも処理
                bias_key = weight_key.replace("weight", "bias")
                if bias_key in left_keys:
                    left_bias = load_tensor(left_model_path, bias_key)
                    cache_bias_path = os.path.join(self.cache_dir, f"{bias_key}.pt")
                    torch.save(left_bias, cache_bias_path)
                    logger.debug(
                        f"左モデルのバイアスをキャッシュに保存しました: {cache_bias_path}"
                    )
                    del left_bias
                    torch.cuda.empty_cache()

        logger.debug("WeightMatchingStrategy の preprocess が完了しました。")
        return (
            {},
            self.cache_dir,
        )  # 左モデルは変更しないため空の辞書、右モデルはキャッシュディレクトリのパスを返す


def get_preprocessing_strategy(strategy_name: str) -> PreprocessingStrategy:
    if strategy_name == "noop":
        return NoOpPreprocessingStrategy()
    elif strategy_name == "weight_matching":
        return WeightMatchingStrategy()
    else:
        raise ValueError(f"未知の前処理戦略: {strategy_name}")
