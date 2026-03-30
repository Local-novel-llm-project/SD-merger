"""ユーティリティモジュール。

モデルの読み込み・保存、YAML 設定ファイルの読み込み、
ファイル名生成などの共通ユーティリティ関数を提供する。
"""

import json
from datetime import datetime
import logging
import os
from collections.abc import Mapping
from typing import Any, Dict, Iterator

from safetensors.torch import save_file, load_file
from safetensors import safe_open
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
            loaded = yaml.safe_load(file)
            return {} if loaded is None else loaded
    except Exception as e:
        logging.error(f"設定ファイルの読み込みに失敗しました: {e}")
        raise


def _normalize_model_path(model_path: str) -> str:
    """モデルパスに .safetensors 拡張子を付加する（二重付加を防止）。
    ただし、すでに .pt や .ckpt などの拡張子を持っている場合はそのまま返す。

    Args:
        model_path: モデルのパス。

    Returns:
        正規化されたパス。
    """
    if any(model_path.endswith(ext) for ext in [".safetensors", ".pt", ".ckpt", ".bin"]):
        return model_path
    return f"{model_path}.safetensors"


def _build_output_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def _build_model_initials(model_name: str) -> str:
    basename = os.path.splitext(os.path.basename(model_name))[0]
    words = [word for word in basename.split("_") if word]
    initials = "".join(word[:3] for word in words)
    return initials or "model"


class LazySafetensorsDict(Mapping[str, Any]):
    """単一の safetensors ファイルからテンソルを遅延読み込みする辞書クラス。"""

    def __init__(self, path):
        self.f = safe_open(path, framework="pt", device="cpu")
        self._keys = self.f.keys()

    def keys(self) -> list[str]:
        return self._keys

    def items(self) -> Iterator[tuple[str, Any]]:
        for k in self._keys:
            yield k, self.f.get_tensor(k)

    def values(self) -> Iterator[Any]:
        for k in self._keys:
            yield self.f.get_tensor(k)

    def __getitem__(self, key):
        return self.f.get_tensor(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys


class ShardedLazySafetensorsDict(Mapping[str, Any]):
    """分割された safetensors ファイル群からテンソルを遅延読み込みする辞書クラス。"""

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.base_dir = os.path.dirname(index_path)
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        self.weight_map = index_data.get("weight_map", {})
        self._keys = list(self.weight_map.keys())

    def keys(self) -> list[str]:
        return self._keys

    def items(self) -> Iterator[tuple[str, Any]]:
        for k in self._keys:
            yield k, self[k]

    def values(self) -> Iterator[Any]:
        for k in self._keys:
            yield self[k]

    def __getitem__(self, key: str):
        if key not in self.weight_map:
            raise KeyError(key)
        filename = self.weight_map[key]
        file_path = os.path.join(self.base_dir, filename)
        # 都度開閉することでファイルディスクリプタの枯渇を防ぐ
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self.weight_map


def load_model(
    model_path: str, use_sdxl_keys: bool | None = None, lazy_load: bool = True
) -> SDKeyWrapper:
    """safetensors 形式のモデルを読み込み、SDKeyWrapper でラップして返す。
    モデルと同じディレクトリにある `config.json` も読み込む。

    Args:
        model_path: モデルファイルのパス、または分割モデルの index.json のパス、もしくはそれを含むディレクトリ。
        use_sdxl_keys: SDXL 形式のキーを使用するかどうか。
            None の場合はモデルのキーから自動判定する。
        lazy_load: テンソルを遅延読み込み（Lazy Load）するかどうか。False の場合はメモリ上に全て展開する。

    Returns:
        読み込まれたモデルの SDKeyWrapper。

    Raises:
        FileNotFoundError: モデルファイルが存在しない場合。
    """
    try:
        console.log(f"[bold green]モデルを読み込んでいます: {model_path}[/bold green]")
        raw: Mapping[str, Any]

        # モデルパスと config.json の探索
        resolved_path = model_path
        config_path = None
        if os.path.isdir(model_path):
            # ディレクトリの場合、index.json を探す
            potential_index = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(potential_index):
                resolved_path = potential_index
            config_path = os.path.join(model_path, "config.json")
        elif os.path.isfile(model_path):
            # ファイルの場合、同じディレクトリの config.json を探す
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

        config = None
        if config_path and os.path.exists(config_path):
            console.log(f"[bold green]設定ファイルを読み込んでいます: {config_path}[/bold green]")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        # 分割モデル (Sharded Model) かどうかの判定
        is_sharded = False
        if os.path.isdir(resolved_path):
            # 再度チェック (ディレクトリが直接渡された場合)
            potential_index = os.path.join(resolved_path, "model.safetensors.index.json")
            if os.path.exists(potential_index):
                is_sharded = True
                index_path = potential_index
        elif resolved_path.endswith(".json"):
            is_sharded = True
            index_path = resolved_path

        if is_sharded:
            if lazy_load:
                raw = ShardedLazySafetensorsDict(index_path)
            else:
                raw_state: dict[str, Any] = {}
                base_dir = os.path.dirname(index_path)
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                unique_files = set(weight_map.values())
                for filename in unique_files:
                    file_path = os.path.join(base_dir, filename)
                    file_tensors = load_file(file_path, device="cpu")
                    raw_state.update(file_tensors)
                raw = raw_state
        else:
            model_file_path = _normalize_model_path(resolved_path)
            # safetensors ではない場合は mmap を有効にして torch.load
            if not model_file_path.endswith(".safetensors"):
                raw = torch.load(
                    model_file_path, map_location="cpu", mmap=lazy_load, weights_only=True
                )
                if "state_dict" in raw:
                    raw = raw["state_dict"]
            else:
                if lazy_load:
                    raw = LazySafetensorsDict(model_file_path)
                else:
                    raw = load_file(model_file_path, device="cpu")

        # use_sdxl_keys が未指定の場合、モデル自体の形式から自動判定
        if use_sdxl_keys is None:
            is_xl = any(k.startswith("conditioner.embedders.0.") for k in raw.keys())
            effective_use_sdxl = is_xl
        else:
            effective_use_sdxl = use_sdxl_keys

        return SDKeyWrapper(raw, effective_use_sdxl, config=config)
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
    left_initials = _build_model_initials(left_model_name)
    right_initials = _build_model_initials(right_model_name)
    timestamp = _build_output_timestamp()
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
