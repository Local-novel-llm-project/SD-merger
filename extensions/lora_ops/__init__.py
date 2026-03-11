import os
import sys
import logging
from types import SimpleNamespace

from module.extension_manager import register_pre_config_hook


def run_lora_operations(config: dict) -> dict:
    """
    設定ファイルの lora_ops セクションを処理するフック。
    実行後、通常のモデルマージを行わない場合は models リストを空にする等の制御が可能。
    """
    lora_ops = config.get("lora_ops")
    if not lora_ops:
        return config

    logging.info("LoRA関連の操作を開始します...")

    # 実行パスに kohyas を追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kohyas_dir = os.path.join(current_dir, "kohyas")
    if kohyas_dir not in sys.path:
        sys.path.insert(0, kohyas_dir)

    operations = lora_ops.get("operations", [])
    last_output_path = None

    for op in operations:
        op_type = op.get("type")
        if op_type == "extract":
            last_output_path = _run_extract_lora(op)
        elif op_type in {"merge", "apply"}:
            last_output_path = _run_merge_lora(op)
        else:
            logging.warning(f"不明なLoRA操作タイプ: {op_type}")

    if operations and lora_ops.get("stop_after_lora_ops", True):
        logging.info(
            "lora_opsが完了しました。stop_after_lora_opsがTrueのため、マージ処理をスキップします。"
        )
        config["_skip_merge"] = True
        config["_skip_merge_output"] = last_output_path

    return config


def _run_extract_lora(op: dict):
    from .kohyas.extract_lora_from_models import svd

    logging.info("LoRA抽出(Extract)を実行します...")
    args = SimpleNamespace(
        v2=op.get("v2", False),
        sdxl=op.get("sdxl", False),
        save_precision=op.get("save_precision", "float"),
        model_org=op["base_model"],
        model_tuned=op["tuned_model"],
        save_to=op["output"],
        dim=op.get("dim", 128),
        v_parameterization=None,
        conv_dim=op.get("conv_dim", None),
        alpha=op.get("alpha", 1.0),
        beta=op.get("beta", 1.0),
        device=op.get("device", "cpu"),
    )
    svd(args)
    logging.info(f"LoRA抽出が完了しました -> {op['output']}")
    return op["output"]


def _select_merge_runner(is_sdxl: bool):
    if is_sdxl:
        from .kohyas.sdxl_merge_lora import merge

        return merge

    from .kohyas.merge_lora import merge

    return merge


def _run_merge_lora(op: dict):
    merge = _select_merge_runner(op.get("sdxl", False))
    is_checkpoint_merge = bool(op.get("sd_model"))

    if is_checkpoint_merge:
        logging.info("LoRAをモデルへマージします...")
    else:
        logging.info("LoRA同士のマージ(Merge)を実行します...")

    args = SimpleNamespace(
        models=op["models"],
        ratios=op.get("ratios", [1.0] * len(op["models"])),
        sd_model=op.get("sd_model"),
        save_to=op["output"],
        precision=op.get("precision", "float"),
        save_precision=op.get("save_precision", "float"),
        sdxl=op.get("sdxl", False),
        v2=op.get("v2", False),
        concat=op.get("concat", False),
        shuffle=op.get("shuffle", False),
        no_metadata=op.get("no_metadata", False),
    )
    merge(args)
    if is_checkpoint_merge:
        logging.info(f"LoRAのモデル適用が完了しました -> {op['output']}")
    else:
        logging.info(f"LoRAマージが完了しました -> {op['output']}")
    return op["output"]


def setup():
    register_pre_config_hook(run_lora_operations)
