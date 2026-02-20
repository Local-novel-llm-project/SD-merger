"""ユーティリティモジュール。

モデルの読み込み・保存、YAML 設定ファイルの読み込み、
ファイル名生成などの共通ユーティリティ関数を提供する。
"""

from datetime import datetime
import logging
import os
from typing import Any, Dict

from safetensors.torch import load_file, save_file
from rich.console import Console
import torch
import yaml

from module.const import SDKeyWrapper

console = Console()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """YAML 設定ファイルを読み込む。

    Args:
        config_path: 設定ファイルのパス。

    Returns:
        設定内容を表す辞書。

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合。
        yaml.YAMLError: YAML のパースに失敗した場合。
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            console.log(f"[bold green]設定ファイルを読み込んでいます: {config_path}[/bold green]")
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise


def _normalize_model_path(model_path: str) -> str:
    """モデルパスに .safetensors 拡張子を付加する（二重付加を防止）。

    Args:
        model_path: モデルのパス。

    Returns:
        .safetensors 拡張子付きのパス。
    """
    if model_path.endswith(".safetensors"):
        return model_path
    return f"{model_path}.safetensors"


def load_model(model_path: str, use_sdxl_keys: bool = None) -> SDKeyWrapper:
    """safetensors 形式のモデルを読み込み、SDKeyWrapper でラップして返す。

    Args:
        model_path: モデルファイルのパス。
        use_sdxl_keys: SDXL 形式のキーを使用するかどうか。
            None の場合はモデルのキーから自動判定する。

    Returns:
        読み込まれたモデルの SDKeyWrapper。

    Raises:
        FileNotFoundError: モデルファイルが存在しない場合。
    """
    model_path = _normalize_model_path(model_path)
    try:
        console.log(f"[bold green]モデルを読み込んでいます: {model_path}[/bold green]")
        raw = load_file(model_path)
        # use_sdxl_keys が未指定の場合、モデル自体の形式から自動判定
        if use_sdxl_keys is None:
            is_xl = any(k.startswith("conditioner.embedders.0.") for k in raw.keys())
            effective_use_sdxl = is_xl
        else:
            effective_use_sdxl = use_sdxl_keys
        return SDKeyWrapper(raw, effective_use_sdxl)
    except Exception as e:
        logging.error(f"{model_path} からモデルの読み込みに失敗しました: {e}")
        raise


def save_model(model: Dict[str, torch.Tensor], model_path: str) -> None:
    """モデルを safetensors 形式で保存する。

    Args:
        model: 保存するモデルの state_dict。
        model_path: 保存先のファイルパス。

    Raises:
        IOError: ファイルの書き込みに失敗した場合。
    """
    try:
        console.log(f"[bold green]モデルを保存しています: {model_path}[/bold green]")
        save_file(model, model_path)
    except Exception as e:
        logging.error(f"{model_path} へのモデルの保存に失敗しました: {e}")
        raise


def generate_filename(left_model_name: str, right_model_name: str) -> str:
    """マージ結果のファイル名を生成する。

    左右のモデル名の各単語から先頭3文字を取り、タイムスタンプを付加する。

    Args:
        left_model_name: 左側モデルのベース名。
        right_model_name: 右側モデルのベース名。

    Returns:
        生成されたファイル名（.safetensors 拡張子付き）。
    """
    left_initials = "".join([word[:3] for word in left_model_name.split("_")])
    right_initials = "".join([word[:3] for word in right_model_name.split("_")])
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{left_initials}_{right_initials}_{timestamp}.safetensors"


def save_processed_key(cache_dir: str, key: str, tensor: torch.Tensor):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.pt")
    torch.save(tensor.cpu(), cache_path)


def load_processed_keys(cache_dir: str) -> Dict[str, torch.Tensor]:
    processed_keys = {}
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.endswith(".pt"):
                key = filename[:-3]
                cache_path = os.path.join(cache_dir, filename)
                tensor = torch.load(cache_path, map_location="cpu")
                processed_keys[key] = tensor
    return processed_keys
