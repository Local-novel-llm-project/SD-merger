"""SD モデルのキー変換ラッパーモジュール。

SD 1.x 系と SDXL 系のキー名プレフィックスの差異を吸収するための
ラッパークラスを提供する。
"""

from typing import Any, Dict


class SDKeyWrapper(dict):
    """SD モデルの state_dict をラップし、キー名の変換を行うクラス。

    SD 1.x 系の `cond_stage_model.` プレフィックスと
    SDXL 系の `conditioner.embedders.0.` プレフィックスの相互変換を行う。

    Args:
        d: モデルの state_dict。
        use_sdxl_keys: SDXL 形式のキーを使用するかどうか。
    """

    _SD1X_PREFIX = "cond_stage_model."
    _SDXL_PREFIX = "conditioner.embedders.0."

    def __init__(self, d: Dict[str, Any], use_sdxl_keys: bool = True):
        self.is_xl = any(k.startswith(self._SDXL_PREFIX) for k in d.keys())

        if use_sdxl_keys and not self.is_xl:
            d = self._convert_keys(d, self._SD1X_PREFIX, self._SDXL_PREFIX)
        elif self.is_xl and not use_sdxl_keys:
            d = self._convert_keys(d, self._SDXL_PREFIX, self._SD1X_PREFIX)

        super().__init__(d)
        self.use_sdxl_keys = use_sdxl_keys

    @staticmethod
    def _convert_keys(d: Dict[str, Any], from_prefix: str, to_prefix: str) -> Dict[str, Any]:
        """キーのプレフィックスを変換する。元の辞書は変更しない。

        Args:
            d: 変換対象の辞書。
            from_prefix: 変換元プレフィックス。
            to_prefix: 変換先プレフィックス。

        Returns:
            プレフィックスが変換された新しい辞書。
        """
        result = {}
        for k, v in d.items():
            if k.startswith(from_prefix):
                result[k.replace(from_prefix, to_prefix, 1)] = v
            else:
                result[k] = v
        return result
