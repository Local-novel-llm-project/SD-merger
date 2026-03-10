import logging
from functools import partial

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from module.bayesian_optimizer.optimiser import Optimiser


class TPEOptimiser(Optimiser):
    """
    Tree-structured Parzen Estimator (TPE) を用いた最適化 (hyperoptライブラリ使用)
    """

    def _target_function(self, params):
        res = self.sd_target_function(**params)
        return {
            "loss": -res,  # hyperopt minimizes loss, so we negate the score
            "status": STATUS_OK,
            "params": params,
        }

    def optimise(self) -> None:
        bounds = self.init_params()
        space = {p: hp.uniform(p, *b) for p, b in bounds.items()}

        self.trials = Trials()
        init_points = self.cfg.get("init_points", 5)
        n_iters = self.cfg.get("n_iters", 20)

        logging.info(
            f"Starting TPE Optimization (init: {init_points}, iters: {n_iters})..."
        )

        tpe._default_n_startup_jobs = init_points
        algo = partial(tpe.suggest, n_startup_jobs=init_points)

        fmin(
            self._target_function,
            space=space,
            algo=algo,
            trials=self.trials,
            max_evals=init_points + n_iters,
            rstate=None,  # Or use a specific random state
        )

    def postprocess(self) -> None:
        logging.info("\n--- TPE Optimization Finished ---")
        scores = []
        for i, res in enumerate(self.trials.losses()):
            # loss is negated score
            score = -res
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

        # 最終モデル保存
        if self.cfg.get("save_best", False):
            logging.info("Saving best model...")
            from main import run_merge_pipeline

            merge_config = self.build_merge_config(
                base_alpha,
                weights,
                save_model=True,
                output_name=self.cfg.get("output_name", "best_merged_model_tpe"),
            )
            run_merge_pipeline(merge_config)
