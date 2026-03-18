import os
import sys
import logging
import importlib
import json
from types import SimpleNamespace
import types

from module.lora_input import normalize_lora_models_and_ratios
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

    _ensure_kohya_import_aliases()

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


def _register_kohya_namespace(name: str, path: str) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [path]
        module.__package__ = name
        sys.modules[name] = module
        return

    existing_paths = list(getattr(module, "__path__", []))
    if path not in existing_paths:
        existing_paths.append(path)
        module.__path__ = existing_paths


def _build_minimum_network_metadata(
    v2,
    base_model,
    network_module,
    network_dim,
    network_alpha,
    network_args,
):
    if network_args is None:
        serialized_network_args = "{}"
    elif isinstance(network_args, str):
        serialized_network_args = network_args
    else:
        serialized_network_args = json.dumps(network_args)

    return {
        "ss_v2": str(bool(v2)),
        "ss_base_model_version": base_model or "",
        "ss_network_module": network_module,
        "ss_network_dim": str(network_dim),
        "ss_network_alpha": str(network_alpha),
        "ss_network_args": serialized_network_args,
    }


def _ensure_kohya_train_util_compat(train_util_module, sai_model_spec_module) -> None:
    if not hasattr(train_util_module, "load_metadata_from_safetensors"):
        train_util_module.load_metadata_from_safetensors = (
            sai_model_spec_module.load_metadata_from_safetensors
        )
    if not hasattr(train_util_module, "SS_METADATA_KEY_V2"):
        train_util_module.SS_METADATA_KEY_V2 = "ss_v2"
    if not hasattr(train_util_module, "SS_METADATA_KEY_BASE_MODEL_VERSION"):
        train_util_module.SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
    if not hasattr(train_util_module, "build_minimum_network_metadata"):
        train_util_module.build_minimum_network_metadata = (
            _build_minimum_network_metadata
        )


def _ensure_kohya_import_aliases() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kohyas_dir = os.path.join(current_dir, "kohyas")

    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if kohyas_dir not in sys.path:
        sys.path.insert(0, kohyas_dir)

    _register_kohya_namespace("scripts", current_dir)
    _register_kohya_namespace("scripts.kohyas", kohyas_dir)
    _register_kohya_namespace("library", kohyas_dir)

    for train_util_name, sai_model_spec_name in [
        ("scripts.kohyas.train_util", "scripts.kohyas.sai_model_spec"),
        ("library.train_util", "library.sai_model_spec"),
        (f"{__package__}.kohyas.train_util", f"{__package__}.kohyas.sai_model_spec"),
    ]:
        train_util_module = importlib.import_module(train_util_name)
        sai_model_spec_module = importlib.import_module(sai_model_spec_name)
        _ensure_kohya_train_util_compat(
            train_util_module,
            sai_model_spec_module,
        )

    return kohyas_dir


def _load_kohya_symbol(module_name: str, symbol_name: str):
    _ensure_kohya_import_aliases()
    module = importlib.import_module(
        f".kohyas.{module_name}",
        package=__package__,
    )
    return getattr(module, symbol_name)


def _run_extract_lora(op: dict):
    svd = _load_kohya_symbol("extract_lora_from_models", "svd")
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
        return _load_kohya_symbol("sdxl_merge_lora", "merge")
    return _load_kohya_symbol("merge_lora", "merge")


def _run_merge_lora(op: dict):
    merge = _select_merge_runner(op.get("sdxl", False))
    is_checkpoint_merge = bool(op.get("sd_model"))
    model_paths, ratios = normalize_lora_models_and_ratios(
        op.get("models"),
        op.get("ratios"),
    )

    if is_checkpoint_merge:
        logging.info("LoRAをモデルへマージします...")
    else:
        logging.info("LoRA同士のマージ(Merge)を実行します...")

    args = SimpleNamespace(
        models=model_paths,
        ratios=ratios,
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
