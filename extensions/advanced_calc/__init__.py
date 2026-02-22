import json
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
from module.extension_manager import register_strategy, register_target_strategy

_CACHE_PATTERNS = {}


def _is_key_matched(key: str, key_patterns_json: str) -> bool:
    if not key_patterns_json or key_patterns_json == "[]":
        return True
    if key_patterns_json not in _CACHE_PATTERNS:
        _CACHE_PATTERNS[key_patterns_json] = json.loads(key_patterns_json)
    patterns = _CACHE_PATTERNS[key_patterns_json]
    return any(p in key for p in patterns)


@merge_method
def target_angle(
    target: Parameter(Tensor),
    diff_l: Parameter(Tensor),
    diff_r: Parameter(Tensor),
    left: Parameter(Tensor),
    right: Parameter(Tensor),
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """角度ベースのマージストラテジー"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return target
    eps = 1e-6
    norm_l = torch.linalg.norm(diff_l)
    norm_r = torch.linalg.norm(diff_r)

    # 角度（コサイン類似度）計算
    theta = (diff_l * diff_r).sum() / torch.clamp(norm_l * norm_r, min=eps)

    # サポートされているテンソルの形状にマッチさせる
    if theta.dim() == 0 and target.dim() > 0:
        theta = theta.unsqueeze(-1)

    t = (2.0 * torch.cos(theta)) / (1.0 + torch.cos(theta))
    avg = (left + right) * 0.5
    res = target * (1.0 - t) + avg * t
    return res


@merge_method
def target_train_difference(
    target: Parameter(Tensor),
    diff_l: Parameter(Tensor),
    diff_r: Parameter(Tensor),
    left: Parameter(Tensor),
    right: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """trainDifference: 差分を疑似的に学習させるマージ手法
    diff_l: target - right (A1)
    diff_r: left - right (AB)
    """
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return target

    diff_AB = left - right
    distance_A0 = torch.abs(diff_AB)
    distance_A1 = torch.abs(left - target)

    sum_distances = distance_A0 + distance_A1
    scale = torch.where(
        sum_distances != 0,
        distance_A1 / sum_distances,
        torch.tensor(0.0, dtype=target.dtype, device=target.device),
    )
    sign_scale = torch.sign(diff_AB)
    scale = sign_scale * torch.abs(scale)

    new_diff = scale * torch.abs(diff_AB)
    # 元実装に合わせ 1.8 を掛ける
    return target + (new_diff * (velocity * 1.8))


@merge_method
def strategy_tensor(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """テンソル交換 (dim=0) ストラテジー
    ※ current_beta は現状 0 固定とする (sd-mechaの制限上引数が1つ)
    """
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return a

    alpha = float(velocity.item() if isinstance(velocity, Tensor) else velocity)
    res = a.clone()

    if res.dim() == 0:
        return res

    talphas = 0
    talphae = int(res.shape[0] * alpha)

    if talphae > talphas:
        res[talphas:talphae] = b[talphas:talphae].clone()

    return res


@merge_method
def strategy_tensor2(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """テンソル交換 (dim=1) ストラテジー"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return a

    alpha = float(velocity.item() if isinstance(velocity, Tensor) else velocity)
    res = a.clone()

    if res.dim() < 2:
        return strategy_tensor(a, b, velocity, key_patterns_json, **kwargs)

    talphas = 0
    talphae = int(res.shape[1] * alpha)

    if talphae > talphas:
        res[:, talphas:talphae] = b[:, talphas:talphae].clone()

    return res


@merge_method
def strategy_extract(
    target: Parameter(Tensor),
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """Extract (モデル用): 共通部分と非共通部分をマージ。
    target: base, a: model_a, b: model_b
    """
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return target

    alpha = float(velocity.item() if isinstance(velocity, Tensor) else velocity)
    # デフォルトの beta=1.0, gamma=1.0 相当として計算
    beta = 1.0
    gamma = 1.0

    a_diff = a.float() - target.float()
    b_diff = b.float() - target.float()

    # calc cosine similarity
    c = torch.nn.functional.cosine_similarity(a_diff, b_diff, dim=-1)
    if c.dim() == 0:
        c = c.unsqueeze(-1)
    c = c.clamp(-1, 1).unsqueeze(-1)
    d = ((c + 1) / 2) ** gamma

    def lerp(x, y, weight):
        return x + weight * (y - x)

    result = target.float() + lerp(a_diff, b_diff, alpha) * lerp(d, 1 - d, beta)
    return result.to(target.dtype)


@merge_method
def strategy_smooth_subtract(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """smoothAdd 用の差分計算: (a - b) に Median と Gaussian フィルタを適用"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return torch.zeros_like(a)

    diff = (a - b).float()

    try:
        import scipy.ndimage
        import numpy as np

        diff_np = diff.cpu().numpy()
        if diff_np.ndim >= 1 and diff_np.size >= 3:
            # Apply median filter
            diff_np = scipy.ndimage.median_filter(diff_np, size=3)
            # Apply Gaussian filter
            diff_np = scipy.ndimage.gaussian_filter(diff_np, sigma=1)

        diff = torch.tensor(diff_np, device=a.device, dtype=a.dtype)
    except ImportError:
        import logging

        logging.warning(
            "scipy がインストールされていないため、smoothAdd のフィルタ処理はスキップされます。(pip install scipy)"
        )

    return diff * velocity


@merge_method
def strategy_cosine_a(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """cosineA (簡易版): テンソル単位で cosine similarity と magnitude_similarity を求め、重みを調整。"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return a

    alpha = float(velocity.item() if isinstance(velocity, Tensor) else velocity)

    # Normalize vectors
    eps = 1e-8
    norm_a = torch.nn.functional.normalize(a.float(), p=2, dim=0)
    norm_b = torch.nn.functional.normalize(b.float(), p=2, dim=0)

    simab = torch.nn.functional.cosine_similarity(
        norm_a.flatten(), norm_b.flatten(), dim=0
    )

    dot_product = torch.dot(norm_a.flatten(), norm_b.flatten())
    mag_a = torch.norm(norm_a)
    mag_b = torch.norm(norm_b)
    magnitude_similarity = dot_product / (mag_a * mag_b + eps)

    combined_similarity = (simab + magnitude_similarity) / 2.0

    # 本家はモデル全体のシミュラリティを min/max で正規化するが、簡易的に自己の類似度を使う
    k = combined_similarity - abs(alpha)
    k = k.clamp(min=0.0, max=1.0)

    # a: theta_0 (base), b: theta_1 (add). We want to favor A
    return b * (1 - k) + a * k


@merge_method
def strategy_cosine_b(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """cosineB (簡易版)"""
    if not _is_key_matched(kwargs.get("key", ""), key_patterns_json):
        return a

    alpha = float(velocity.item() if isinstance(velocity, Tensor) else velocity)

    eps = 1e-8
    simab = torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    )

    dot_product = torch.dot(a.flatten().float(), b.flatten().float())
    mag_a = torch.norm(a.float())
    mag_b = torch.norm(b.float())
    magnitude_similarity = dot_product / (mag_a * mag_b + eps)

    combined_similarity = (simab + magnitude_similarity) / 2.0

    k = combined_similarity - abs(alpha)
    k = k.clamp(min=0.0, max=1.0)

    return b * (1 - k) + a * k


def setup():
    register_target_strategy("angle", target_angle)
    register_target_strategy("trainDifference", target_train_difference)
    register_target_strategy("extract", strategy_extract)

    # 差分/ミックス戦略（AとBを混ぜるタイプ）
    register_strategy("tensor", strategy_tensor)
    register_strategy("tensor2", strategy_tensor2)
    register_strategy("smoothAdd", strategy_smooth_subtract)
    register_strategy("cosineA", strategy_cosine_a)
    register_strategy("cosineB", strategy_cosine_b)
