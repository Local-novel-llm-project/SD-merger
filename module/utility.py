import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable

import torch
import yaml
from rich.console import Console
from safetensors import safe_open
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


def process_model_tensors(
    model_path: str,
    process_function: Callable[[str, torch.Tensor], None],
    key_patterns: Iterable[str] = None,
):
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key_patterns is None or any(k in key for k in key_patterns):
                tensor = f.get_tensor(key)
                # 必要に応じてテンソルをGPUに転送
                tensor = tensor.to("cuda" if torch.cuda.is_available() else "cpu")
                # テンソルに処理を適用
                process_function(key, tensor)
                # メモリを解放
                del tensor
                torch.cuda.empty_cache()


def load_tensor(model_path: str, key: str) -> torch.Tensor:
    with safe_open(model_path, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    return tensor


def save_tensor(tensor: torch.Tensor, cache_dir: str, key: str):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.pt")
    torch.save(tensor.cpu(), cache_path)


def load_cached_tensors(cache_dir: str) -> Dict[str, torch.Tensor]:
    model = {}
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pt"):
            key = filename[:-3]
            cache_path = os.path.join(cache_dir, filename)
            tensor = torch.load(cache_path, map_location="cpu")
            model[key] = tensor
    return model


def process_and_cache_tensor(
    model_path: str, cache_dir: str, process_function, key_patterns=None
):
    os.makedirs(cache_dir, exist_ok=True)

    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        if key_patterns is not None:
            keys = [k for k in keys if any(pattern in k for pattern in key_patterns)]

        for key in keys:
            tensor = f.get_tensor(key)
            tensor = tensor.to("cuda" if torch.cuda.is_available() else "cpu")
            processed_tensor = process_function(key, tensor)
            cache_path = os.path.join(cache_dir, f"{key}.pt")
            torch.save(processed_tensor.cpu(), cache_path)
            del tensor, processed_tensor
            torch.cuda.empty_cache()


def assemble_model_from_cache(cache_dir: str) -> Dict[str, torch.Tensor]:
    model = {}
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pt"):
            key = filename[:-3]
            cache_path = os.path.join(cache_dir, filename)
            tensor = torch.load(cache_path, map_location="cpu")
            model[key] = tensor
    return model
