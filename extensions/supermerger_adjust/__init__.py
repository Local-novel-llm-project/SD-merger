import json
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
from module.extension_manager import register_pre_merge_hook

# COLS calculation specific for Adjust
COLS = [[-1, 1 / 3, 2 / 3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
COLSXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]

FINETUNES = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
    "model.diffusion_model.out.2.weight",
    "model.diffusion_model.out.2.bias",
]


def colorcalc(cols: list, isxl: bool) -> list:
    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i, x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]


def fineman(fine: list, isxl: bool) -> list:
    """fine(8つの要素のリスト)を内部計算用の6つのリストに変換する"""
    if not fine or len(fine) != 8:
        return None

    # detail/noise/contrast/brightness/color
    processed_fine = [
        1 - fine[0] * 0.01,
        1 + fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1 + fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3] * 0.02] + colorcalc(fine[4:8], isxl),
    ]
    return processed_fine


@merge_method
def adjust_tune_node(
    tensor: Parameter(Tensor),
    fines_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """
    Adjust Node 処理。
    計算をストリーミングで行うため、受け取った Tensor(key付き) に対して
    Adjustの設定に基づくスケール/バイアス演算を適用して返す。
    """
    key = kwargs.get("key", "")

    if key not in FINETUNES:
        return tensor

    fines = json.loads(fines_json)
    if not fines:
        return tensor

    index = FINETUNES.index(key)

    if index < 5:
        # scale
        scale = float(fines[index])
        return tensor * scale
    else:
        # bias (tensor addition)
        # fines[5] is a list of 4 values (for the 4 channels of out.2.bias)
        bias_values = fines[5]
        # SD/SDXL output channels is 4. Add the bias values to the 4 channels.
        if len(bias_values) == 4 and tensor.dim() >= 1 and tensor.shape[0] == 4:
            bias_tensor = torch.tensor(
                bias_values, dtype=tensor.dtype, device=tensor.device
            )
            # handle multidimensional if needed, but bias is usually 1D
            view_shape = [-1] + [1] * (tensor.dim() - 1)
            return tensor + bias_tensor.view(*view_shape)
        else:
            return tensor

    fines = json.loads(fines_json)
    if not fines:
        return tensor

    index = FINETUNES.index(key)

    if index < 5:
        # scale
        scale = float(fines[index])
        return tensor * scale
    else:
        # bias (tensor addition)
        # fines[5] is a list of 4 values (for the 4 channels of out.2.bias)
        bias_values = fines[5]
        # check if shapes match (SD/SDXL output channels is 4)
        if len(bias_values) == 4 and tensor.dim() >= 1 and tensor.shape[0] == 4:
            bias_tensor = torch.tensor(
                bias_values, dtype=tensor.dtype, device=tensor.device
            )
            # handle multidimensional if needed, but bias is usually 1D
            if tensor.dim() > 1:
                bias_tensor = bias_tensor.view(-1, *([1] * (tensor.dim() - 1)))
            return tensor + bias_tensor
        else:
            return tensor


def apply_adjust_hook(config: dict, recipe):
    """
    yaml設定の中に 'adjust' セクションがあれば、レシピをラップして返すフック。

    YAML設定例:
    adjust:
      values: [0, 0, 0, 0, 0, 0, 0, 0] # 8つの数値
      is_xl: false
    """
    adjust_config = config.get("adjust", None)
    if not adjust_config:
        return recipe

    values = adjust_config.get("values", [0, 0, 0, 0, 0, 0, 0, 0])
    is_xl = adjust_config.get("is_xl", False)

    # 互換性のため文字列のカンマ区切りでもリストでも受け付ける
    if isinstance(values, str):
        values = [float(v.strip()) for v in values.split(",")]

    if len(values) < 8:
        values = values + [0] * (8 - len(values))
    elif len(values) > 8:
        values = values[:8]

    fines = fineman(values, is_xl)
    if not fines:
        return recipe

    # JSON化して sd-mecha ノードのパラメータとして渡す
    import logging

    logging.info(f"Adjust 機能を適用します (is_xl={is_xl})...")
    return adjust_tune_node(recipe, fines_json=json.dumps(fines))


def setup():
    register_pre_merge_hook(apply_adjust_hook)
