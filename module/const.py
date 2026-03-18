"""SD モデルのキー変換ラッパーモジュール。

SD 1.x 系と SDXL 系のキー名プレフィックスの差異を吸収するための
ラッパークラスを提供する。
"""

from typing import Any, Dict, Iterator
from collections.abc import Mapping


class SDKeyWrapper(dict, Mapping):
    """SD モデルの state_dict をラップし、キー名の変換を行うクラス。

    SD 1.x 系の `cond_stage_model.` プレフィックスと
    SDXL 系の `conditioner.embedders.0.` プレフィックスの相互変換を行う。

    Args:
        d: モデルの state_dict。
        use_sdxl_keys: SDXL 形式のキーを使用するかどうか。
    """

    _SD1X_PREFIX = "cond_stage_model."
    _SDXL_PREFIX = "conditioner.embedders.0."

    def __init__(self, d: Mapping[str, Any], use_sdxl_keys: bool = True):
        self._d = d
        self.is_xl = any(k.startswith(self._SDXL_PREFIX) for k in d.keys())
        self.use_sdxl_keys = use_sdxl_keys

        self._needs_conversion = False
        self._from_prefix = ""
        self._to_prefix = ""

        if use_sdxl_keys and not self.is_xl:
            self._needs_conversion = True
            self._from_prefix = self._SD1X_PREFIX
            self._to_prefix = self._SDXL_PREFIX
        elif self.is_xl and not use_sdxl_keys:
            self._needs_conversion = True
            self._from_prefix = self._SDXL_PREFIX
            self._to_prefix = self._SD1X_PREFIX

    def _convert_key(self, k: str) -> str:
        if not self._needs_conversion:
            return k
        if k.startswith(self._from_prefix):
            return k.replace(self._from_prefix, self._to_prefix, 1)
        return k

    def _revert_key(self, k: str) -> str:
        if not self._needs_conversion:
            return k
        if k.startswith(self._to_prefix):
            return k.replace(self._to_prefix, self._from_prefix, 1)
        return k

    def __getitem__(self, key: str) -> Any:
        original_key = self._revert_key(key)
        return self._d[original_key]

    def __iter__(self) -> Iterator[str]:
        for k in self._d.keys():
            yield self._convert_key(k)

    def __len__(self) -> int:
        return len(self._d)

    def keys(self) -> list:  # type: ignore
        return [self._convert_key(k) for k in self._d.keys()]

    def items(self):  # type: ignore
        for k in self._d.keys():
            ck = self._convert_key(k)
            yield ck, self[ck]

    def values(self):  # type: ignore
        for k in self._d.keys():
            ck = self._convert_key(k)
            yield self[ck]
