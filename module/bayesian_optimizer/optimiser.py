import json
import logging
import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List

from module.bayesian_optimizer.bounds import Bounds
from module.bayesian_optimizer.scorer import AestheticScorer

# SD-mergerの内部モジュールのインポート（target functionで呼び出すため）
from main import run_merge_pipeline
from module.generation import generate_image


class Optimiser:
    """
    マージの最適化エンジンの基底クラス
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.bounds_initialiser = Bounds()
        self.scorer = AestheticScorer(
            method=self.cfg.get("scorer_method", "laion"), device=self.cfg.get("device", "cuda")
        )
        self.iteration = 0
        self.best_rolling_score = 0.0

        self.output_dir = Path(self.cfg.get("output_dir", "./output/bayesian"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_name = f"optim_{int(time.time())}"
        self.best_log_path = self.output_dir / "best.log"

    def init_params(self) -> Dict:
        """探索空間を取得する"""
        return self.bounds_initialiser.get_bounds(
            frozen_params=self.cfg.get("frozen_params", {}),
            custom_ranges=self.cfg.get("custom_ranges", {}),
            groups=self.cfg.get("groups", []),
        )

    def sd_target_function(self, **params) -> float:
        """
        オプティマイザが呼び出す目的関数
        1. 指定されたパラメータでマージ実行
        2. 画像生成
        3. スコアリング
        4. スコア平均を返す
        """
        self.iteration += 1
        is_warmup = self.iteration <= self.cfg.get("init_points", 5)
        phase = "warmup" if is_warmup else "optimisation"

        logging.info(f"\n--- {phase} Iteration {self.iteration} ---")

        # 1. パラメータの組み立て
        weights, base_alpha = self.bounds_initialiser.assemble_params(
            params=params,
            frozen=self.cfg.get("frozen_params", {}),
            groups=self.cfg.get("groups", []),
        )

        # 2. マージ実行
        # mbwの25値をモデル1, モデル2の重みとして適用
        # SD-mergerの `strategy: mbw_each` を使用
        merge_config = {
            "mode": "weight_sum",
            "models": [
                {
                    "left": self.cfg["model_a"],
                    "right": self.cfg["model_b"],
                    "strategy": "mbw_each",
                    "base_alpha": base_alpha,
                    "mbw": weights,
                }
            ],
            # 最適化中は中間モデルをメモリ上に保持する（ファイル出力しない）
            "save_model": False,
            "device": self.cfg.get("device", "cuda"),
        }

        logging.info(f"Merging models with base_alpha={base_alpha:.4f}")

        # 内部API呼び出し（テンソル辞書を返す想定）
        try:
            merged_state = run_merge_pipeline(merge_config)
            if not merged_state:
                logging.error("Merge output is empty.")
                return 0.0
        except Exception as e:
            logging.error(f"Merge error: {e}")
            return 0.0

        # 3. 画像生成
        images = []
        try:
            # generate_image がメモリ上のテンソル辞書を受け取れるように拡張されている前提
            for prompt in self.cfg.get("prompts", [""]):
                gen_result = generate_image(
                    model_path_or_dict=merged_state,
                    prompt=prompt,
                    negative_prompt=self.cfg.get("negative_prompt", ""),
                    width=self.cfg.get("width", 512),
                    height=self.cfg.get("height", 512),
                    num_inference_steps=self.cfg.get("steps", 20),
                    guidance_scale=self.cfg.get("cfg_scale", 7.0),
                    seed=self.cfg.get("seed", -1),
                    device=self.cfg.get("device", "cuda"),
                )
                if gen_result.get("image"):
                    images.append(gen_result["image"])

                # VRAM開放
                if "pipe" in gen_result:
                    del gen_result["pipe"]
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return 0.0

        # 生成後は一時ファイルを削除
        if merged_state and os.path.exists(merged_state):
            try:
                os.remove(merged_state)
                logging.info(f"Deleted temp model file: {merged_state}")
            except Exception as e:
                logging.error(f"Failed to delete temp model file: {e}")

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not images:
            return 0.0

        # 4. スコアリング
        avg_score = self.scorer.batch_score(images)
        logging.info(f"Score: {avg_score:.4f}")

        # ベストスコアの更新
        if avg_score > self.best_rolling_score:
            logging.info("⭐ NEW BEST SCORE!")
            self.best_rolling_score = avg_score
            self.save_best_log(base_alpha, weights, avg_score)

            # 画像保存
            if self.cfg.get("save_imgs", True):
                for i, img in enumerate(images):
                    img_path = self.output_dir / f"best-{self.iteration}-{i}.png"
                    img.save(img_path)

        return avg_score

    @abstractmethod
    def optimise(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")

    def save_best_log(self, base_alpha: float, weights: List[float], score: float) -> None:
        log_data = {"score": score, "base_alpha": base_alpha, "mbw": weights, "iteration": self.iteration}
        with open(self.best_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
        logging.info(f"Saved best.log to {self.best_log_path}")
