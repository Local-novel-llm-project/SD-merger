"""Arthemy Live Tuner 機能の再実装モジュール (sd-mecha版)。

sd-mecha のレシピノードとして組み込める遅延評価 (Lazy Evaluation) 対応の Tuner です。
ストリーミングマージの最終段として各テンソルごとのスケール倍率計算ノードを適用します。
"""

import re
import json
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
from module.extension_manager import register_pre_merge_hook


def _get_target_weight_soft(w: float) -> float:
    # Soft Value のカーネルロジック (CLIP用)
    return 0.8 + (0.2 * w)


def _get_target_weight_model_soft(w: float) -> float:
    # Soft Value のカーネルロジック (UNet用)
    if w <= 1.0:
        return max(0.0, -1.02 * (w**2) + 2.02 * w)
    return 1.0 + (w - 1.0) * 0.133


# --- Configs & Maps ---
BLOCK_BOUNDARIES_CLIP = {"syntax_end": 0.35, "semantic_end": 0.75}
TOTAL_LAYERS_SDXL = 32

GROUP_MAP_UNET = {
    "IN_Layout_Geometry": [0, 1],
    "IN_Perspective_Masses": [2, 3],
    "IN_Subject_Identity": [4, 5],
    "IN_Global_Composition": [6, 7, 8],
    "MID_Core_Concept": [9],
    "OUT_Art_Style_Medium": [10, 11],
    "OUT_Material_Substance": [12],
    "OUT_Lighting_Atmosphere": [13, 14],
    "OUT_Shadows_Depth": [15],
    "OUT_Texture_Details": [16, 17],
    "OUT_Final_Sharpness": [18],
}

_CACHE = {}


@merge_method
def arthemy_tune_node(
    tensor: Parameter(Tensor),
    mode: Parameter(str) = "Soft Value",
    clip_config_json: Parameter(str) = "{}",
    unet_config_json: Parameter(str) = "{}",
    **kwargs,
) -> Return(Tensor):
    """
    Arthemy Tuner Node 処理。
    計算をストリーミングで行うため、受け取った Tensor(key付き) に対して
    CLIP/UNet のブロックごとに計算した倍率を乗算して返す。
    設定値は JSON 文字列として受け取り Node 内でデコードし最適化キャッシュする。
    """
    key = kwargs.get("key", "")

    if not ("text_model" in key or "model.diffusion_model" in key):
        return tensor

    # 設定のパース/前計算のキャッシュ (テンソル毎の処理負荷削減)
    cache_key = (mode, clip_config_json, unet_config_json)
    if cache_key not in _CACHE:
        clip_conf = json.loads(clip_config_json)
        unet_conf = json.loads(unet_config_json)

        get_w_clip = _get_target_weight_soft if mode == "Soft Value" else lambda x: x
        w_base_c = get_w_clip(clip_conf.get("base_scale", 1.0))
        w_syntax = get_w_clip(clip_conf.get("syntax_rigidity", 1.0))
        w_semantic = get_w_clip(clip_conf.get("semantic_focus", 1.0))
        w_style = get_w_clip(clip_conf.get("style_abstraction", 1.0))
        clip_data = (w_base_c, w_syntax, w_semantic, w_style)

        get_w_unet = (
            _get_target_weight_model_soft if mode == "Soft Value" else lambda x: x
        )
        final_weights = [1.0] * 19
        c_dict = unet_conf.get("config_dict", {})
        for group_name, indices in GROUP_MAP_UNET.items():
            s_val = c_dict.get(group_name, 1.0)
            r_val = get_w_unet(s_val)
            for idx in indices:
                if 0 <= idx < 19:
                    final_weights[idx] = r_val
        weights_map = {}
        for i in range(9):
            weights_map[f"IN_{i}"] = final_weights[i]
        weights_map["MID"] = final_weights[9]
        for i in range(9):
            weights_map[f"OUT_{i}"] = final_weights[10 + i]
        w_base_u = get_w_unet(unet_conf.get("base_scale", 1.0))
        unet_data = (weights_map, w_base_u)

        _CACHE[cache_key] = (clip_data, unet_data, bool(clip_conf), bool(unet_conf))

    clip_data, unet_data, has_clip, has_unet = _CACHE[cache_key]

    # --- CLIP Tuner Logic ---
    if "text_model" in key and not (
        key.endswith(".position_ids") or key.endswith(".logit_scale")
    ):
        if has_clip:
            w_base_c, w_syntax, w_semantic, w_style = clip_data
            target_scale = w_base_c
            match = re.search(r"\.layers\.(\d+)\.", key)
            if match:
                ratio = int(match.group(1)) / TOTAL_LAYERS_SDXL
                if ratio <= BLOCK_BOUNDARIES_CLIP["syntax_end"]:
                    target_scale = w_syntax
                elif ratio <= BLOCK_BOUNDARIES_CLIP["semantic_end"]:
                    target_scale = w_semantic
                else:
                    target_scale = w_style
            if target_scale != 1.0:
                return tensor * target_scale

    # --- UNet Tuner Logic ---
    if "model.diffusion_model" in key:
        if has_unet:
            weights_map, w_base_u = unet_data
            target_weight = w_base_u
            if "input_blocks" in key:
                match = re.search(r"input_blocks\.(\d+)\.", key)
                if match and int(match.group(1)) <= 8:
                    target_weight = weights_map.get(
                        f"IN_{int(match.group(1))}", w_base_u
                    )
            elif "middle_block" in key:
                target_weight = weights_map.get("MID", w_base_u)
            elif "output_blocks" in key:
                match = re.search(r"output_blocks\.(\d+)\.", key)
                if match and int(match.group(1)) <= 8:
                    target_weight = weights_map.get(
                        f"OUT_{int(match.group(1))}", w_base_u
                    )
            if target_weight != 1.0:
                return tensor * target_weight

    return tensor


def apply_arthemy_tuner_recipe(recipe, arthemy_config: dict):
    """
    sd-mecha のレシピノードに対し、Arthemy Tuner を計算グラフの最終段として適用する。

    Args:
        recipe: sd-mecha のマージレシピグラフ (merge_out_node)。
        arthemy_config: yaml等で読み込んだ Arthemy Tuner 設定辞書。

    Returns:
        Arthemy Tuner ノードが重畳された新しい sd-mecha レシピ。
    """
    if not arthemy_config:
        return recipe

    mode = arthemy_config.get("mode", "Soft Value")
    clip_config = arthemy_config.get("clip", {})
    unet_config = arthemy_config.get("unet", {})
    # unet設定にネストされた config_dict を含める処理
    unet_json_obj = {
        "base_scale": unet_config.get("base_scale", 1.0),
        "config_dict": unet_config,
    }

    # レシピノードをラップして返す
    return arthemy_tune_node(
        recipe,
        mode=mode,
        clip_config_json=json.dumps(clip_config),
        unet_config_json=json.dumps(unet_json_obj),
    )


def apply_arthemy_tuner_hook(config: dict, recipe):
    arthemy_config = config.get("arthemy_tuner")
    if arthemy_config:
        import logging

        logging.info("Arthemy Tuner 拡張機能: 合成グラフに組み込みます...")
        return apply_arthemy_tuner_recipe(recipe, arthemy_config)
    return recipe


def setup():
    register_pre_merge_hook(apply_arthemy_tuner_hook)
