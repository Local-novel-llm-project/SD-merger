import os
import sys
import json
import logging
import argparse
from typing import Sequence

import torch

# Patch torch.load to default to mmap=True to prevent RAM spikes
# when sd_mecha or other scripts read PyTorch pickle files (.pt / .ckpt).
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("mmap", True)
    kwargs.setdefault("weights_only", True)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import sd_mecha
from rich.console import Console

from module.logging_config import logger
from module.exceptions import ConfigError, SDMergerError
from module.config_schema import MergeConfig
from pydantic import ValidationError

from module.calc_method import get_calculation_strategy
from module.calc_target import (
    get_normalization_calculation_strategy,
    get_target_calculation_strategy,
)
from module.arthemy_tuner_config import (
    ARTHEMY_TUNER_MODES,
    CLIP_FIELD_SPECS,
    UNET_SECTION_SPECS,
    build_arthemy_tune_job_config,
)
from module.utility import generate_filename, load_yaml_config
from module.extension_manager import (
    load_extensions,
    run_pre_config_hooks,
    run_pre_merge_hooks,
    run_post_merge_hooks,
)

# logger configuration is handled by module.logging_config

console = Console()


@sd_mecha.merge_method
def scale_tensor(
    a: sd_mecha.Parameter(torch.Tensor), scale: sd_mecha.Parameter(torch.Tensor) = 1.0
) -> sd_mecha.Return(torch.Tensor):
    return a * scale


def _iter_unet_field_specs():
    for section in UNET_SECTION_SPECS:
        for field in section["fields"]:
            yield field


def _field_name_to_option(name: str) -> str:
    return "--" + name.lower().replace("_", "-")


def _configure_debug_mode(debug_enabled: bool) -> None:
    if not debug_enabled:
        return
    logger.setLevel(logging.DEBUG)
    sd_mecha.set_log_level(logging.DEBUG)


def _ensure_extensions_loaded() -> None:
    load_extensions()


def _get_sd_mecha_merge_log_level() -> int:
    if logger.isEnabledFor(logging.DEBUG):
        return logging.DEBUG
    return logging.INFO


def _add_debug_argument(parser: argparse.ArgumentParser, *, default: object = False) -> None:
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=default,
        help="DEBUGログレベルを有効にする",
    )


def _build_initial_recipe(config: dict):
    target_model_path = config.get("target_model")
    if not target_model_path:
        return None, None, None

    lazy_load = config.get("lazy_load", True)
    from module.utility import load_model

    wrapper = load_model(target_model_path, lazy_load=lazy_load)
    model_dict = wrapper._d
    return sd_mecha.model(model_dict), wrapper.config, target_model_path


def _validate_merge_inputs(models: list[dict], recipe) -> None:
    if models:
        return
    if recipe is not None:
        logger.info("models が未指定のため、target_model に対して pre-merge 拡張のみを適用します。")
        return

    logger.error("設定ファイルにモデルが指定されていません。")
    raise ConfigError("設定ファイルにモデルが指定されていません。")


def _resolve_model_strategy_name(
    model_config: dict,
    key: str,
    default: str,
) -> str:
    value = model_config.get(key)
    if value is None:
        return default
    return str(value)


def _ensure_mapping_config(raw_config: object) -> dict:
    if isinstance(raw_config, dict):
        return raw_config
    raise ConfigError("設定ファイルの最上位はマッピング形式である必要があります。")


def _resolve_effective_output_dir(
    config: dict,
    default_output_dir: str,
    *,
    explicit_output_dir: bool,
    validated_default_output_dir: str,
) -> str:
    configured_output_dir = config.get("output_dir")
    if explicit_output_dir:
        return str(configured_output_dir or default_output_dir)
    if configured_output_dir and configured_output_dir != validated_default_output_dir:
        return str(configured_output_dir)
    return default_output_dir


def _resolve_output_filename(
    config: dict, models: list[dict], target_model_path: str | None
) -> str:
    output_filename = config.get("output_name")
    if output_filename:
        return output_filename

    if models:
        first_left_name = os.path.basename(models[0]["left"])
        last_right_name = os.path.basename(models[-1]["right"])
        return generate_filename(first_left_name, last_right_name)

    if target_model_path:
        target_name = os.path.splitext(os.path.basename(target_model_path))[0]
        return generate_filename(target_name, "tuned")

    raise ConfigError("出力ファイル名を決定できませんでした。")


def _resolve_output_path(
    config: dict,
    default_output_dir: str,
    models: list[dict],
    target_model_path: str | None,
) -> tuple[bool, str]:
    save_model = config.get("save_model", True)
    if not save_model:
        import tempfile

        fd, output_path = tempfile.mkstemp(suffix=".safetensors", prefix="sd_merge_tmp_")
        os.close(fd)
        return False, output_path

    output_filename = _resolve_output_filename(config, models, target_model_path)
    sharded_output = config.get("sharded_output", False)

    if sharded_output:
        # For sharded output, the "path" is a directory.
        output_path = os.path.join(default_output_dir, os.path.splitext(output_filename)[0])
        os.makedirs(output_path, exist_ok=True)
    else:
        # For single file output, the path includes the filename.
        output_path = os.path.join(default_output_dir, output_filename)
        os.makedirs(default_output_dir, exist_ok=True)

    return True, output_path


def _build_clip_overrides_from_args(args: argparse.Namespace) -> dict[str, float]:
    overrides = {}
    for spec in CLIP_FIELD_SPECS:
        attr_name = "clip_base_scale" if spec["name"] == "base_scale" else spec["name"]
        value = getattr(args, attr_name, None)
        if value is not None:
            overrides[spec["name"]] = value
    return overrides


def _build_unet_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "unet_base_scale", None) is not None:
        overrides["base_scale"] = args.unet_base_scale

    for spec in _iter_unet_field_specs():
        value = getattr(args, spec["name"], None)
        if value is not None:
            overrides[spec["name"]] = value

    vectors_override = getattr(args, "vectors_override", None)
    if vectors_override:
        overrides["vectors_override"] = vectors_override

    return overrides


def _build_tune_config_from_args(args: argparse.Namespace) -> dict:
    return build_arthemy_tune_job_config(
        target_model=args.model,
        mode=args.mode,
        clip_overrides=_build_clip_overrides_from_args(args),
        unet_overrides=_build_unet_overrides_from_args(args),
        output_name=args.output_name,
        save_model=not getattr(args, "no_save", False),
    )


def _add_merge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", type=str, default="sd_config.yaml", help="設定ファイルのパス"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./merged", help="出力ディレクトリのパス"
    )
    parser.add_argument(
        "--no-lazy",
        action="store_true",
        help="Lazy Load (遅延読み込み) を無効にし、全モデルをメモリに読み込む",
    )


def _add_tune_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="チューニング対象のモデルファイルパス。",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./merged",
        help="出力ディレクトリのパス",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="出力ファイル名。未指定時は自動生成。",
    )
    parser.add_argument(
        "--mode",
        choices=ARTHEMY_TUNER_MODES,
        default="Soft Value",
        help="Arthemy Tuner の重み付けモード。",
    )
    parser.add_argument(
        "--clip-base-scale",
        dest="clip_base_scale",
        type=float,
        default=None,
        help="CLIP 全体へ適用する基本倍率。",
    )
    parser.add_argument(
        "--unet-base-scale",
        dest="unet_base_scale",
        type=float,
        default=None,
        help="UNet 全体へ適用する基本倍率。",
    )
    parser.add_argument(
        "--vectors-override",
        type=str,
        default=None,
        help="UNet 19ブロックを直接指定するカンマ区切り文字列。",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="保存せず一時ファイルへ出力する。",
    )

    for spec in CLIP_FIELD_SPECS:
        if spec["name"] == "base_scale":
            continue
        parser.add_argument(
            _field_name_to_option(spec["name"]),
            dest=spec["name"],
            type=float,
            default=None,
            help=spec["description"],
        )

    for spec in _iter_unet_field_specs():
        parser.add_argument(
            _field_name_to_option(spec["name"]),
            dest=spec["name"],
            type=float,
            default=None,
            help=spec["description"],
        )


def _add_ui_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="0.0.0.0", help="UI の bind host")
    parser.add_argument("--port", type=int, default=7860, help="UI の listen port")
    parser.add_argument("--share", action="store_true", help="Gradio share を有効化")


def _create_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="モデルの差分計算とマージツール (sd-mecha版)")
    _add_merge_arguments(parser)
    _add_debug_argument(parser)
    return parser


def _create_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="モデルの差分計算とマージツール (sd-mecha版)")
    _add_debug_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="YAML 設定ファイルでマージを実行")
    _add_merge_arguments(merge_parser)
    _add_debug_argument(merge_parser, default=argparse.SUPPRESS)

    tune_parser = subparsers.add_parser("tune", help="Arthemy Tuner を単一モデルへ適用")
    _add_tune_arguments(tune_parser)
    _add_debug_argument(tune_parser, default=argparse.SUPPRESS)

    ui_parser = subparsers.add_parser("ui", help="Gradio UI を起動")
    _add_ui_arguments(ui_parser)
    _add_debug_argument(ui_parser, default=argparse.SUPPRESS)

    return parser


def _find_subcommand(argv: Sequence[str]) -> str | None:
    subcommands = {"merge", "tune", "ui"}
    for token in argv:
        if token in {"-d", "--debug"}:
            continue
        if token in subcommands:
            return token
        return None
    return None


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(argv or sys.argv[1:])
    if _find_subcommand(argv):
        return _create_subcommand_parser().parse_args(argv)

    args = _create_legacy_parser().parse_args(argv)
    args.command = "merge"
    return args


def _run_merge_command(args: argparse.Namespace) -> str | None:
    _ensure_extensions_loaded()
    raw_config = load_yaml_config(args.config)
    if getattr(args, "no_lazy", False):
        raw_config["lazy_load"] = False
    return run_merge_pipeline(raw_config, default_output_dir=args.output)


def _run_tune_command(args: argparse.Namespace) -> str | None:
    _ensure_extensions_loaded()
    config = _build_tune_config_from_args(args)
    output_path = run_merge_pipeline(config, default_output_dir=args.output)
    if output_path:
        logger.info(f"Arthemy Tuner output saved to {output_path}")
    return output_path


def _run_ui_command(args: argparse.Namespace) -> int:
    from ui.app import launch_ui

    _ensure_extensions_loaded()
    launch_ui(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def _dispatch_cli_command(args: argparse.Namespace) -> int:
    _configure_debug_mode(bool(getattr(args, "debug", False)))

    if args.command == "merge":
        _run_merge_command(args)
        return 0
    if args.command == "tune":
        _run_tune_command(args)
        return 0
    if args.command == "ui":
        return _run_ui_command(args)

    raise ConfigError(f"未知のコマンドです: {args.command}")


def run_merge_pipeline(raw_config: dict, default_output_dir: str = "./merged") -> str | None:
    """設定辞書を受け取り、マージ処理を実行する。
    戻り値: マージされたモデルのファイルパス。save_model が False の場合は一時ファイルのパス。
    """
    normalized_raw_config = _ensure_mapping_config(raw_config)

    try:
        validated_config = MergeConfig(**normalized_raw_config)
        config = run_pre_config_hooks(validated_config.model_dump())
    except ValidationError as e:
        logger.error(f"コンフィグのバリデーションエラー: {e}")
        raise ConfigError("Invalid configuration syntax or types.", original_error=e)

    if config.get("_skip_merge"):
        return config.get("_skip_merge_output")
    
    dtype = config.get("dtype", "float16")
    dtype = getattr(torch, dtype)

    recipe, final_config, target_model_path = _build_initial_recipe(config)
    models = config.get("models", [])
    _validate_merge_inputs(models, recipe)

    for model_config in models:
        lazy_load = config.get("lazy_load", True)
        from module.utility import load_model

        left_wrapper = load_model(model_config["left"], lazy_load=lazy_load)
        if final_config is None:
            final_config = left_wrapper.config
        left_dict = left_wrapper._d

        right_wrapper = load_model(model_config["right"], lazy_load=lazy_load)
        right_dict = right_wrapper._d

        left_node = sd_mecha.model(left_dict)
        right_node = sd_mecha.model(right_dict)
        target_velocity = model_config.get("velocity", 1.0)
        left_right_velocity = model_config.get("left_right_velocity", 1.0)
        strategy_name = _resolve_model_strategy_name(
            model_config,
            "strategy",
            "addition",
        )
        key_patterns = model_config.get("key_patterns", None)
        replace_with = model_config.get("replace_with", None)
        target_strategy_name = _resolve_model_strategy_name(
            model_config,
            "target_strategy",
            "addition",
        )
        normalization_strategy_name = _resolve_model_strategy_name(
            model_config,
            "normalization_strategy",
            "none",
        )

        if not key_patterns:
            if recipe is None:
                logger.error(
                    "target_model と key_patterns の両方が未指定です。どちらかを指定してください。"
                )
                raise ConfigError("target_model と key_patterns の両方が未指定です。")
            logger.error('key_patterns の指定は必須です。(全キーを指定する場合は "." 等を指定)')
            raise ConfigError("key_patterns の指定は必須です。")

        calc_func = get_calculation_strategy(strategy_name, replace_with)
        target_func = get_target_calculation_strategy(target_strategy_name)
        norm_func = get_normalization_calculation_strategy(normalization_strategy_name)

        patterns_json = json.dumps(key_patterns)

        diff_node = calc_func(
            left_node,
            right_node,
            velocity=left_right_velocity,
            key_patterns_json=patterns_json,
        )

        if recipe is not None:
            if target_strategy_name == "angle":
                diff_l = calc_func(
                    left_node,
                    recipe,
                    velocity=left_right_velocity,
                    key_patterns_json=patterns_json,
                )
                diff_r = calc_func(
                    right_node,
                    recipe,
                    velocity=left_right_velocity,
                    key_patterns_json=patterns_json,
                )
                merged = target_func(
                    recipe,
                    diff_l=diff_l,
                    diff_r=diff_r,
                    left=left_node,
                    right=right_node,
                    key_patterns_json=patterns_json,
                )
            else:
                merged = target_func(
                    recipe,
                    diff_node,
                    velocity=target_velocity,
                    key_patterns_json=patterns_json,
                )

            if norm_func is not None:
                merged = norm_func(recipe, merged)

            recipe = merged
        else:
            recipe = scale_tensor(diff_node, scale=target_velocity)

    effective_output_dir = _resolve_effective_output_dir(
        config,
        default_output_dir,
        explicit_output_dir="output_dir" in normalized_raw_config,
        validated_default_output_dir=validated_config.output_dir,
    )

    save_model, output_path = _resolve_output_path(
        config, effective_output_dir, models, target_model_path
    )

    recipe = run_pre_merge_hooks(config, recipe)

    sd_mecha.set_log_level(_get_sd_mecha_merge_log_level())

    sharded_output = config.get("sharded_output", False)
    if sharded_output:
        logger.info(f"マージ処理を実行し、{output_path} に sharded 形式で保存します...")
        logger.info("sd-mecha がインメモリマージを開始します。")
        try:
            # The output=None tells sd-mecha to return the merged state_dict in memory
            state_dict = sd_mecha.merge(recipe, output_dtype=dtype, output=None)
        except Exception as e:
            logger.error(f"sd-mecha merging error: {e}")
            from module.exceptions import MergeError

            raise MergeError("Merge failed during sd_mecha processing", original_error=e)

        logger.info("インメモリマージが完了しました。")
        logger.info("sharded 形式での保存を開始します。")

        from huggingface_hub.serialization._torch import save_torch_state_dict

        save_torch_state_dict(
            state_dict,
            save_directory=output_path,
            max_shard_size=config.get("max_shard_size", "5GB"),
        )

    else:
        logger.info(f"マージ処理を実行し、{output_path} に保存します...")
        logger.info("sd-mecha がストリーミング処理を開始します。")
        try:
            sd_mecha.merge(recipe, output_dtype=dtype, output=output_path)
        except Exception as e:
            logger.error(f"sd-mecha merging error: {e}")
            from module.exceptions import MergeError

            raise MergeError("Merge failed during sd_mecha processing", original_error=e)

    logger.info("マージが完了しました。")

    if save_model and final_config:
        config_output_path = (
            os.path.join(output_path, "config.json")
            if sharded_output
            else os.path.join(os.path.dirname(output_path), "config.json")
        )
        logger.info(f"設定ファイルを保存しています: {config_output_path}")
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=2)

    if save_model:
        run_post_merge_hooks(config, output_path)

    return output_path


def run_from_config_file(config_path: str, output_dir: str) -> str | None:
    """メイン処理。設定ファイルに従いモデルのマージを sd-mecha レシピとして構築して実行する。

    Args:
        config_path: YAML 設定ファイルのパス。
        output_dir: 出力ディレクトリのパス。
    """
    raw_config = load_yaml_config(config_path)
    return run_merge_pipeline(raw_config, default_output_dir=output_dir)


def main(config_path: str, output_dir: str) -> str | None:
    return run_from_config_file(config_path, output_dir)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    return _dispatch_cli_command(args)


if __name__ == "__main__":
    try:
        sys.exit(cli_main())
    except SDMergerError as e:
        logger.error(f"Application Error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error occurred.")
        sys.exit(1)
