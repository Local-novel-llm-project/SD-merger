import os
import sys
import logging
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

    for op in operations:
        op_type = op.get("type")
        if op_type == "extract":
            _run_extract_lora(op)
        elif op_type == "merge":
            _run_merge_lora(op)
        elif op_type == "apply":
            logging.info(
                "LoRAのCheckpointへの適用は、通常のmodels設定(strategy: replace)等を使用してsd-mechaで実行してください。"
            )
        else:
            logging.warning(f"不明なLoRA操作タイプ: {op_type}")

    # LoRA操作のみを行う場合、models設定を空にして終了させるか確認
    if config.get("models") and lora_ops.get("stop_after_lora_ops", True):
        logging.info(
            "lora_opsが完了しました。stop_after_lora_opsがTrueのため、マージ処理をスキップします。"
        )
        config["models"] = []

    return config


def _run_extract_lora(op: dict):
    from .kohyas.extract_lora_from_models import svd
    from types import SimpleNamespace

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


def _run_merge_lora(op: dict):
    from .kohyas.merge_lora import merge
    from types import SimpleNamespace

    logging.info("LoRA同士のマージ(Merge)を実行します...")
    args = SimpleNamespace(
        models=op["models"],
        ratios=op.get("ratios", [1.0] * len(op["models"])),
        save_to=op["output"],
        precision=op.get("precision", "float"),
        save_precision=op.get("save_precision", "float"),
        sdxl=op.get("sdxl", False),
        v2=op.get("v2", False),
        concat=op.get("concat", False),
        shuffle=op.get("shuffle", False),
    )
    merge(args)
    logging.info(f"LoRAマージが完了しました -> {op['output']}")


def setup():
    register_pre_config_hook(run_lora_operations)
