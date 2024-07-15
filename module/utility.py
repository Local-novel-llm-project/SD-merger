from datetime import datetime
import logging
from typing import Any, Dict
from safetensors.torch import load_file, save_file
from rich.console import Console
import torch
import yaml

from module.const import SDKeyWrapper

console = Console()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            console.log(
                f"[bold green]設定ファイルを読み込んでいます: {config_path}[/bold green]"
            )
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise


def load_model(model_path: str, use_sdxl_keys=None) -> SDKeyWrapper:
    try:
        console.log(f"[bold green]モデルを読み込んでいます: {model_path}[/bold green]")
        tensor = SDKeyWrapper(load_file(model_path), True)
        if use_sdxl_keys is None:
            tensor.use_sdxl_keys = tensor.is_xl
        else:
            tensor.use_sdxl_keys = use_sdxl_keys
        return tensor
    except Exception as e:
        logging.error(f"{model_path} からモデルの読み込みに失敗しました: {e}")
        raise


def save_model(model: Dict[str, torch.Tensor], model_path: str) -> None:
    try:
        console.log(f"[bold green]モデルを保存しています: {model_path}[/bold green]")
        save_file(model, model_path)
    except Exception as e:
        logging.error(f"{model_path} へのモデルの保存に失敗しました: {e}")
        raise


def generate_filename(left_model_name: str, right_model_name: str) -> str:
    left_initials = "".join([word[:3] for word in left_model_name.split("_")])
    right_initials = "".join([word[:3] for word in right_model_name.split("_")])
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{left_initials}_{right_initials}_{timestamp}.safetensors"
