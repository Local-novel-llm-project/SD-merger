from typing import Any, Dict


class SDKeyWrapper(dict):
    def __init__(self, d: Dict[str, Any], use_sdxl_keys=True):
        self.is_xl = any(k.startswith("conditioner.embedders.0.") for k in d.keys())
        if use_sdxl_keys and not self.is_xl:
            kk = list(d.keys())
            for k in kk:
                if k.startswith("cond_stage_model."):
                    d[k.replace("cond_stage_model.", "conditioner.embedders.0.", 1)] = (
                        d.pop(k)
                    )
        elif self.is_xl and not use_sdxl_keys:
            kk = list(d.keys())
            for k in kk:
                if k.startswith("conditioner.embedders.0."):
                    d[k.replace("conditioner.embedders.0.", "cond_stage_model.", 1)] = (
                        d.pop(k)
                    )
        super().__init__(d)
        self.use_sdxl_keys = use_sdxl_keys
