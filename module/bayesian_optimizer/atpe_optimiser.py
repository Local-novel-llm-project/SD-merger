import logging

from hyperopt import STATUS_OK, Trials, atpe, fmin, hp

from module.bayesian_optimizer.optimiser import Optimiser


class ATPEOptimiser(Optimiser):
    """
    Adaptive Tree-structured Parzen Estimator (ATPE) を用いた最適化 (hyperoptライブラリ使用)
    """

    def _target_function(self, params):
        res = self.sd_target_function(**params)
        return {
            "loss": -res + 10,  # hyperopt minimizes loss
            "status": STATUS_OK,
            "params": params,
        }

    def optimise(self) -> None:
        bounds = self.init_params()
        space = {p: hp.uniform(p, *b) for p, b in bounds.items()}

        self.trials = Trials()
        init_points = self.cfg.get("init_points", 5)
        n_iters = self.cfg.get("n_iters", 20)

        logging.info(f"Starting ATPE Optimization (init: {init_points}, iters: {n_iters})...")

        fmin(
            self._target_function,
            space=space,
            algo=atpe.suggest,
            trials=self.trials,
            max_evals=init_points + n_iters,
        )

    def postprocess(self) -> None:
        logging.info("\n--- ATPE Optimization Finished ---")
        scores = []
        for i, res in enumerate(self.trials.losses()):
            # loss was -score + 10, so score = 10 - loss
            score = 10 - res
            scores.append(score)

        best = self.trials.best_trial
        best_params = best["result"]["params"]

        weights, base_alpha = self.bounds_initialiser.assemble_params(
            params=best_params,
            frozen=self.cfg.get("frozen_params", {}),
            groups=self.cfg.get("groups", []),
        )

        logging.info(f"Best Score: {max(scores):.4f}")
        logging.info(f"Best Base Alpha: {base_alpha:.4f}")
        logging.info(f"Best MBW: {weights}")

        # draw_plot(scores)

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
                "output_name": self.cfg.get("output_name", "best_merged_model_atpe"),
                "device": self.cfg.get("device", "cuda"),
            }
            run_merge_pipeline(merge_config)
