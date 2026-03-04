import logging

from bayes_opt import BayesianOptimization, Events
from scipy.stats import qmc

from module.bayesian_optimizer.optimiser import Optimiser


class BayesOptimiser(Optimiser):
    """
    Gaussian Processベースのベイズ最適化（bayesian-optimizationライブラリ使用）
    """

    def optimise(self) -> None:
        pbounds = self.init_params()

        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=self.cfg.get("seed", 1),
            verbose=2,
            # allow_duplicate_points=True
        )

        # ログ登録
        from bayes_opt.logger import JSONLogger

        logger = JSONLogger(path=str(self.output_dir / f"{self.log_name}_bayes.json"))
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        init_points = self.cfg.get("init_points", 5)

        # Latin Hypercube Sampling (LHS)
        if self.cfg.get("latin_hypercube_sampling", False):
            logging.info("Using Latin Hypercube Sampling for initial exploration.")
            sampler = qmc.LatinHypercube(d=len(pbounds))
            samples = sampler.random(init_points)
            l_bounds = [b[0] for b in pbounds.values()]
            u_bounds = [b[1] for b in pbounds.values()]
            scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

            for sample in scaled_samples.tolist():
                params = dict(zip(pbounds, sample))
                self.optimizer.probe(params=params, lazy=True)

            init_points = 0  # LHSでプローブ済みのため

        n_iters = self.cfg.get("n_iters", 20)
        logging.info(f"Starting Bayesian Optimization (init: {init_points}, iters: {n_iters})...")

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iters,
        )

    def postprocess(self) -> None:
        logging.info("\n--- Optimization Finished ---")

        if not hasattr(self, "optimizer") or not self.optimizer.res:
            logging.warning("No results to process.")
            return

        scores = [r["target"] for r in self.optimizer.res]
        best_params = self.optimizer.max["params"]

        weights, base_alpha = self.bounds_initialiser.assemble_params(
            params=best_params,
            frozen=self.cfg.get("frozen_params", {}),
            groups=self.cfg.get("groups", []),
        )

        logging.info(f"Best Score: {max(scores):.4f}")
        logging.info(f"Best Base Alpha: {base_alpha:.4f}")
        logging.info(f"Best MBW: {weights}")

        # TODO: Convergence Plot描画処理への統合
        # draw_plot(scores)

        # 最終モデルを保存する場合
        if self.cfg.get("save_best", False):
            logging.info("Saving best model...")
            from main import run_merge_pipeline

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
                "save_model": True,
                "output_name": self.cfg.get("output_name", "best_merged_model"),
                "device": self.cfg.get("device", "cuda"),
            }
            run_merge_pipeline(merge_config)
