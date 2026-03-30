import warnings
from typing import Dict, List, Optional, Tuple

# SD-mergerのブロック設定を使用
NUM_TOTAL_BLOCKS = 12  # SD-merger uses IN00-11, OUT00-11
# Note: we need to handle IN, MID, OUT differently than the reference which just uses "block_i_alpha" etc.


class Bounds:
    """
    パラメータの探索空間（Bounds）を管理するクラス
    """

    @staticmethod
    def set_block_bounds(block_name: str, lb: float = 0.0, ub: float = 1.0) -> Tuple[float, float]:
        return (lb, ub)

    @staticmethod
    def default_bounds(
        custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        SD-mergerの26ブロック構成に基づくデフォルトの探索空間を生成
        """
        if custom_ranges is None:
            custom_ranges = {}

        block_names = []
        # IN blocks, MID block, OUT blocks
        for i in range(12):
            block_names.append(f"IN{i:02d}")
        block_names.append("MID00")
        for i in range(12):
            block_names.append(f"OUT{i:02d}")

        # Base alpha
        block_names.append("base_alpha")

        ranges = {b: (0.0, 1.0) for b in block_names}
        ranges.update(custom_ranges)

        return {b: Bounds.set_block_bounds(b, *ranges[b]) for b in block_names}

    @staticmethod
    def freeze_bounds(
        bounds: Dict[str, Tuple[float, float]], frozen: Optional[Dict[str, float]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        固定（Freeze）されたパラメータを探索空間から除外する
        """
        if frozen is None:
            return bounds
        return {b: r for b, r in bounds.items() if b not in frozen}

    @staticmethod
    def group_bounds(
        bounds: Dict[str, Tuple[float, float]], groups: Optional[List[List[str]]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        パラメータをグループ化（Group）し、同一の重みとして探索空間を縮小する
        """
        if groups is None:
            return bounds

        for group in groups:
            if not group:
                continue
            ranges = {bounds[b] for b in group if b in bounds}
            if len(ranges) > 1:
                w = (
                    f"different values for the same group: {group}"
                    + f" we're picking {group[0]} range: {bounds[group[0]]}"
                )
                warnings.warn(w)
                group_range = bounds[group[0]]
            elif ranges:
                group_range = ranges.pop()
            else:
                # all frozen
                continue

            group_name = "-".join(group)
            bounds[group_name] = group_range
            for b in group:
                if b in bounds:
                    del bounds[b]
        return bounds

    @staticmethod
    def freeze_groups(bounds: Dict, groups: List[List[str]], frozen: Dict[str, float]) -> Dict:
        """
        グループ内の一部がFreezeされている場合の処理
        """
        if groups is None or frozen is None:
            return bounds

        for group in groups:
            group_name = "-".join(group)
            if group_name in bounds and any(b in frozen for b in group):
                del bounds[group_name]
        return bounds

    @staticmethod
    def get_bounds(
        frozen_params: Optional[Dict[str, float]] = None,
        custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        groups: Optional[List[List[str]]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Freeze, Custom Range, Group をすべて適用した最終的な探索空間を取得
        """
        if frozen_params is None:
            frozen_params = {}
        if custom_ranges is None:
            custom_ranges = {}
        if groups is None:
            groups = []

        bounds = Bounds.default_bounds(custom_ranges)
        not_frozen_bounds = Bounds.freeze_bounds(bounds, frozen_params)
        grouped_bounds = Bounds.group_bounds(not_frozen_bounds, groups)
        return Bounds.freeze_groups(grouped_bounds, groups, frozen_params)

    @staticmethod
    def get_value(
        params: Dict[str, float], block_name: str, frozen: Dict[str, float], groups: List[List[str]]
    ) -> float:
        """
        最適化エンジンから返されたパラメータ（Dictionary）から、特定のブロックの重みを取得する
        """
        if block_name in params:
            return params[block_name]
        if groups is not None:
            for group in groups:
                if block_name in group:
                    group_name = "-".join(group)
                    if group_name in params:
                        return params[group_name]
                    if group[0] in frozen:
                        return frozen[group[0]]
                    if group[0] in params:
                        return params[group[0]]
        return frozen.get(block_name, 0.5)  # fallback 0.5 or better logic

    @staticmethod
    def assemble_params(
        params: Dict[str, float],
        frozen: Optional[Dict[str, float]] = None,
        groups: Optional[List[List[str]]] = None,
    ) -> Tuple[List[float], float]:
        """
        最適化エンジンの出力（パラーメータ）から、SD-merger用の25値リスト(mbw)とbase_alphaを構築する
        """
        if frozen is None:
            frozen = {}
        if groups is None:
            groups = []

        weights = []

        # IN00-11
        for i in range(12):
            block_name = f"IN{i:02d}"
            weights.append(Bounds.get_value(params, block_name, frozen, groups))

        # MID00
        weights.append(Bounds.get_value(params, "MID00", frozen, groups))

        # OUT00-11
        for i in range(12):
            block_name = f"OUT{i:02d}"
            weights.append(Bounds.get_value(params, block_name, frozen, groups))

        assert len(weights) == 25

        base_alpha = Bounds.get_value(params, "base_alpha", frozen, groups)

        return weights, base_alpha
