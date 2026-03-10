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
        from module.extension_manager import load_extensions

        load_extensions()
        self.cfg = cfg
        self.bounds_initialiser = Bounds()
        self.scorer = AestheticScorer(
            method=self.cfg.get("scorer_method", "laion"),
            device=self.cfg.get("device", "cuda"),
        )
        self.iteration = 0
        self.best_rolling_score = 0.0

        self.output_dir = Path(self.cfg.get("output_dir", "./output/bayesian"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_name = f"optim_{int(time.time())}"
        self.best_log_path = self.output_dir / "best.log"

    @staticmethod
    def _format_mbw_values(values: List[float]) -> str:
        return ",".join(f"{value:.6f}" for value in values)

    def build_merge_config(
        self,
        base_alpha: float,
        weights: List[float],
        *,
        save_model: bool,
        output_name: str | None = None,
    ) -> Dict:
        blend_ratios = [base_alpha, *weights]
        complementary_ratios = [1.0 - ratio for ratio in blend_ratios]

        config = {
            "models": [
                {
                    "left": self.cfg["model_a"],
                    "right": self.cfg["model_b"],
                    "strategy": "mbw_each",
                    "mbw_a": self._format_mbw_values(complementary_ratios),
                    "mbw_b": self._format_mbw_values(blend_ratios),
                }
            ],
            "save_model": save_model,
            "device": self.cfg.get("device", "cuda"),
        }
        if output_name:
            config["output_name"] = output_name
        return config

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

        merge_config = self.build_merge_config(base_alpha, weights, save_model=False)

        logging.info(f"Merging models with base_alpha={base_alpha:.4f}")

        merged_model_path = None
        try:
            merged_model_path = run_merge_pipeline(merge_config)
            if not merged_model_path:
                logging.error("Merge output is empty.")
                return 0.0
        except Exception as e:
            logging.error(f"Merge error: {e}")
            return 0.0

        # 3. 画像生成
        images = []
        try:
            for prompt in self.cfg.get("prompts", [""]):
                generated_images = generate_image(
                    model_path=merged_model_path,
                    prompt=prompt,
                    negative_prompt=self.cfg.get("negative_prompt", ""),
                    width=self.cfg.get("width", 512),
                    height=self.cfg.get("height", 512),
                    steps=self.cfg.get("steps", 20),
                    cfg=self.cfg.get("cfg_scale", 7.0),
                    seed=self.cfg.get("seed", -1),
                )
                if generated_images:
                    images.extend(generated_images)
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return 0.0
        finally:
            if merged_model_path and os.path.exists(merged_model_path):
                try:
                    os.remove(merged_model_path)
                    logging.info(f"Deleted temp model file: {merged_model_path}")
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

    def save_best_log(
        self, base_alpha: float, weights: List[float], score: float
    ) -> None:
        log_data = {
            "score": score,
            "base_alpha": base_alpha,
            "mbw": weights,
            "iteration": self.iteration,
        }
        with open(self.best_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
        logging.info(f"Saved best.log to {self.best_log_path}")
