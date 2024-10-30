import logging
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple

import torch
from rich.logging import RichHandler  # 追加
from rich.progress import Progress

from lib.hungarian_algorithm import hungarian_algorithm, lowmem_hungarian_algorithm

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


class PreprocessingStrategy(ABC):
    def __init__(self, progress = None) -> None:
        super().__init__()
        self.progress = progress
        # self.progress_callback = None  # この行をコメントアウトまたは削除

    # def set_progress_callback(
    #     self, progress_callback: Callable[[int], None] = None
    # ) -> None:
    #     self.progress_callback = progress_callback
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['progress']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.progress = None

    @abstractmethod
    def preprocess(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        target_layer_list: Iterable[str],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pass

    # def post_operation(self):
    #     if self.progress_callback is not None:
    #         self.progress_callback()


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

    def __init__(self, progress = None, max_iter: int = 10):
        super().__init__(progress)
        self.max_iter = max_iter

    def preprocess(
        self,
        left_model: Dict[str, torch.Tensor],
        right_model: Dict[str, torch.Tensor],
        target_layer_list: Iterable[str],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        logger.debug("WeightMatchingStrategy の preprocess を開始します。")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"使用デバイス: {device}")

        left_processed = left_model.copy()
        right_processed = right_model.copy()

        weight_keys = [
            k
            for k in left_model.keys()
            if (target_layer_list is None or any(layer in k for layer in target_layer_list)) and "weight" in k
        ]

        # プログレスバーの設定
        # with Progress(console=self.progress_console) as progress:
        if self.progress is not None:
            task = self.progress.add_task("Processing weights...", total=len(weight_keys))

        for weight_key in weight_keys:
            logger.debug(f"処理対象のキー: {weight_key}")
            if weight_key in right_model:
                # テンソルを取得
                left_weight = left_model[weight_key].to(device, dtype=torch.float32)
                right_weight = right_model[weight_key].to(
                    device, dtype=torch.float32
                )

                # 対応するバイアスキー
                bias_key = weight_key.replace("weight", "bias")
                if bias_key in left_model and bias_key in right_model:
                    left_bias = left_model[bias_key].to(device, dtype=torch.float32)
                    right_bias = right_model[bias_key].to(
                        device, dtype=torch.float32
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
                    # if right_bias is not None:
                    #     permuted_right_bias = right_bias[P_indices]
                    # else:
                    #     permuted_right_bias = None

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
                    right_weight_norm = torch.nn.functional.normalize(
                        permuted_right_weight_flat, dim=1, eps=1e-7
                    )

                    # 類似度行列計算
                    similarity = torch.mm(left_weight_norm, right_weight_norm.t())

                    # 無効な値が含まれていないかチェック
                    if (
                        torch.isnan(similarity).any()
                        or torch.isinf(similarity).any()
                    ):
                        logger.error(
                            "類似度行列に無効な値 (NaN または Inf) が含まれています。"
                        )
                        break

                    # コスト行列を生成（最大化問題を最小化問題に変換）
                    cost_matrix = -similarity.detach()

                    # コスト行列のサイズに応じてアルゴリズムを選択
                    if num_units > 8192:  # 閾値は必要に応じて調整してください
                        ans_pos = lowmem_hungarian_algorithm(cost_matrix)
                        logger.debug("lowmem_hungarian_algorithm を使用します。")
                    else:
                        ans_pos = hungarian_algorithm(cost_matrix)
                        logger.debug("hungarian_algorithm を使用します。")

                    col_ind = ans_pos[:, 1]
                    new_P_indices = col_ind

                    # 収束判定
                    if torch.equal(P_indices, new_P_indices):
                        logger.debug("Permutation が収束しました。")
                        break

                    P_indices = new_P_indices
                    logger.debug("Permutation を更新しました。")

                # 最終的なPermutationを適用
                matched_right_weight = right_weight[P_indices].to(
                    device=left_weight.device
                )
                right_processed[weight_key] = matched_right_weight.cpu()

                if right_bias is not None:
                    matched_right_bias = right_bias[P_indices].to(
                        device=left_bias.device
                    )
                    right_processed[bias_key] = matched_right_bias.cpu()

                # GPUメモリからテンソルを削除
                del left_weight, right_weight, matched_right_weight
                if left_bias is not None:
                    del left_bias, right_bias, matched_right_bias
                torch.cuda.empty_cache()

            else:
                logger.warning(
                    f"右モデルにキー {weight_key} が見つからないため、左モデルの重みを使用します。"
                )

            # プログレスバーの更新
            if self.progress is not None:
                self.progress.advance(task)

        logger.debug("WeightMatchingStrategy の preprocess が完了しました。")
        return left_processed, right_processed


def get_preprocessing_strategy(strategy_name: str, progress = None) -> PreprocessingStrategy:
    if strategy_name == "noop":
        return NoOpPreprocessingStrategy(progress)
    elif strategy_name == "weight_matching":
        return WeightMatchingStrategy(progress)
    else:
        raise ValueError(f"未知の前処理戦略: {strategy_name}")
