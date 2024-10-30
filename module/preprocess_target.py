import logging
from abc import ABC, abstractmethod
from typing import Dict, Iterable

import torch
from rich.logging import RichHandler  # 追加
from rich.progress import Progress  # 追加

from lib.hungarian_algorithm import hungarian_algorithm, lowmem_hungarian_algorithm

# ログの設定を更新
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


class TargetPreprocessingStrategy(ABC):
    def __init__(self, progress=None) -> None:
        super().__init__()
        self.progress = progress
        # self.progress_callback = None  # 不要なコールバック関連のコードをコメントアウトまたは削除

    # def set_progress_callback(
    #     self, progress_callback: Callable[[], None] = None
    # ) -> None:
    #     self.progress_callback = progress_callback
    
    # for pickling
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
        target_model: Dict[str, torch.Tensor],
        target_layer_list: Iterable[str],
        reference_model: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pass

    # def post_operation(self):
    #     if self.progress_callback is not None:
    #         self.progress_callback()


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

    def __init__(self, progress = None, max_iter: int = 10):
        super().__init__(progress)
        self.max_iter = max_iter

    def preprocess(
        self,
        target_model: Dict[str, torch.Tensor],
        target_layer_list: Iterable[str],
        reference_model: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logger.debug("WeightMatchingTargetStrategy の preprocess を開始します。")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"使用デバイス: {device}")

        target_processed = target_model.copy()

        weight_keys = [
            k
            for k in target_model.keys()
            if (target_layer_list is None or any(layer in k for layer in target_layer_list)) and "weight" in k
        ]

        # プログレスバーの設定
        # with Progress() as progress:
        if self.progress is not None:
            task = self.progress.add_task("Processing weights...", total=len(weight_keys))

        for weight_key in weight_keys:
            logger.debug(f"処理対象のキー: {weight_key}")
            if weight_key in reference_model:
                # テンソルを取得
                target_weight = target_model[weight_key].to(
                    device, dtype=torch.float32
                )
                reference_weight = reference_model[weight_key].to(
                    device, dtype=torch.float32
                )

                # 対応するバイアスキー
                bias_key = weight_key.replace("weight", "bias")
                if bias_key in target_model and bias_key in reference_model:
                    target_bias = target_model[bias_key].to(
                        device, dtype=torch.float32
                    )
                    reference_bias = reference_model[bias_key].to(
                        device, dtype=torch.float32
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
                matched_target_weight = target_weight[P_indices].to(
                    device=target_weight.device
                )
                target_processed[weight_key] = matched_target_weight.cpu()

                if target_bias is not None:
                    matched_target_bias = target_bias[P_indices].to(
                        device=target_bias.device
                    )
                    target_processed[bias_key] = matched_target_bias.cpu()

                # GPUメモリからテンソルを削除
                del target_weight, reference_weight, matched_target_weight
                if target_bias is not None:
                    del target_bias, reference_bias, matched_target_bias
                torch.cuda.empty_cache()

            else:
                logger.warning(
                    f"参照モデルにキー {weight_key} が見つからないため、ターゲットモデルの重みを使用します。"
                )

            # プログレスバーの更新
            if self.progress is not None:
                self.progress.advance(task)

        logger.debug("WeightMatchingTargetStrategy の preprocess が完了しました。")
        return target_processed


def get_target_preprocessing_strategy(
    strategy_name: str,
    progress = None,
) -> TargetPreprocessingStrategy:
    if strategy_name == "noop":
        return NoOpTargetPreprocessingStrategy(progress)
    elif strategy_name == "weight_matching":
        return WeightMatchingTargetStrategy(progress)
    else:
        raise ValueError(f"未知のターゲット前処理戦略: {strategy_name}")
