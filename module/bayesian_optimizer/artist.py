import os
from typing import List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PathT = os.PathLike


def maxwhere(li: List[float]) -> Tuple[int, float]:
    m = 0
    mi = -1
    for i, v in enumerate(li):
        if v > m:
            m = v
            mi = i
    return mi, m


def minwhere(li: List[float]) -> Tuple[int, float]:
    m = 10
    mi = -1
    for i, v in enumerate(li):
        if v < m:
            m = v
            mi = i
    return mi, m


def convergence_plot(
    scores: List[float],
    figname: PathT = None,
    minimise=False,
) -> None:
    """
    推移グラフ（収束プロット）を出力する
    """
    plt.figure()

    plt.plot(scores)

    # ハイライト（最大/最小点）
    star_i, star_score = minwhere(scores) if minimise else maxwhere(scores)
    plt.plot(star_i, star_score, "or")

    plt.xlabel("iterations")

    if minimise:
        plt.ylabel("loss")
    else:
        plt.ylabel("score (aesthetic)")

    sns.despine()

    if figname:
        plt.title(Path(figname).stem)
        import logging

        logging.info(f"Saving convergence plot to: {figname}")
        plt.tight_layout()
        plt.savefig(figname)
    plt.close()
