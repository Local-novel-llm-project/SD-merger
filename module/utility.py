import logging
import os
import hashlib
import pickle
from datetime import datetime
from typing import Any, Dict

import torch
import yaml
from rich.console import Console
from safetensors.torch import load_file, save_file

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
        if model_path.endswith(".safetensors"):
            tensor = SDKeyWrapper(load_file(model_path), True)
        else:
            tensor = SDKeyWrapper(torch.load(model_path, map_location="cpu", weights_only=True), True)
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
        if model_path.endswith(".safetensors"):
            save_file(model, model_path)
        else:
            torch.save(model, model_path)
    except Exception as e:
        logging.error(f"{model_path} へのモデルの保存に失敗しました: {e}")
        raise


def generate_filename(left_model_name: str, right_model_name: str, ext: str) -> str:
    left_initials = "".join([word[:3] for word in left_model_name.split("_")])
    right_initials = "".join([word[:3] for word in right_model_name.split("_")])
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{left_initials}_{right_initials}_{timestamp}{ext}"


def save_processed_key(cache_dir: str, key: str, tensor: torch.Tensor, prefix: str | None = None) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    if prefix:
        key = f"{prefix}_{key}"
    cache_path = os.path.join(cache_dir, f"{key}.pt")
    torch.save(tensor.cpu(), cache_path)


def load_processed_keys(cache_dir: str, prefix: str | None = None) -> Dict[str, torch.Tensor]:
    processed_keys = {}
    if os.path.exists(cache_dir):
        for filename in [d for d in os.listdir(cache_dir) if prefix is None or d.startswith(prefix+"_")]:
            if filename.endswith(".pt"):
                if prefix:
                    key = filename[len(prefix)+1:-3]
                else:
                    key = filename[:-3]
                cache_path = os.path.join(cache_dir, filename)
                tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
                processed_keys[key] = tensor
    return processed_keys

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def hash_obj(obj: object) -> str:
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()
