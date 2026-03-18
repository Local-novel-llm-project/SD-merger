from __future__ import annotations

from typing import Any


ARTHEMY_TUNER_MODES = ["Soft Value", "Real Value"]

CLIP_FIELD_SPECS = [
    {
        "name": "base_scale",
        "label": "CLIP Base Scale",
        "description": "CLIP 全体へ適用する基本倍率。",
        "default": 1.0,
    },
    {
        "name": "syntax_rigidity",
        "label": "Syntax Rigidity",
        "description": "文法やトークン順序の厳密さ。",
        "default": 1.0,
    },
    {
        "name": "semantic_focus",
        "label": "Semantic Focus",
        "description": "主題や属性の認識の強さ。",
        "default": 1.0,
    },
    {
        "name": "style_abstraction",
        "label": "Style Abstraction",
        "description": "抽象度やスタイル解釈の強さ。",
        "default": 1.0,
    },
]

UNET_SECTION_SPECS = [
    {
        "title": "Input Blocks",
        "fields": [
            {
                "name": "IN_Layout_Geometry",
                "label": "Layout Geometry",
                "description": "高解像度レイアウトと空間の骨格。",
                "default": 1.0,
            },
            {
                "name": "IN_Perspective_Masses",
                "label": "Perspective Masses",
                "description": "大局的な形状とパース。",
                "default": 1.0,
            },
            {
                "name": "IN_Subject_Identity",
                "label": "Subject Identity",
                "description": "被写体の意味と構造。",
                "default": 1.0,
            },
            {
                "name": "IN_Global_Composition",
                "label": "Global Composition",
                "description": "高レベルな意味的構成。",
                "default": 1.0,
            },
        ],
    },
    {
        "title": "Middle Block",
        "fields": [
            {
                "name": "MID_Core_Concept",
                "label": "Core Concept",
                "description": "中核となる意味処理。",
                "default": 1.0,
            },
        ],
    },
    {
        "title": "Output Blocks",
        "fields": [
            {
                "name": "OUT_Art_Style_Medium",
                "label": "Art Style Medium",
                "description": "画風やメディウム感。",
                "default": 1.0,
            },
            {
                "name": "OUT_Material_Substance",
                "label": "Material Substance",
                "description": "材質感や物質感。",
                "default": 1.0,
            },
            {
                "name": "OUT_Lighting_Atmosphere",
                "label": "Lighting Atmosphere",
                "description": "照明と空気感。",
                "default": 1.0,
            },
            {
                "name": "OUT_Shadows_Depth",
                "label": "Shadows Depth",
                "description": "影と奥行き感。",
                "default": 1.0,
            },
            {
                "name": "OUT_Texture_Details",
                "label": "Texture Details",
                "description": "質感や細部の密度。",
                "default": 1.0,
            },
            {
                "name": "OUT_Final_Sharpness",
                "label": "Final Sharpness",
                "description": "最終的なシャープネス。",
                "default": 1.0,
            },
        ],
    },
]


def _collect_defaults(specs: list[dict[str, Any]]) -> dict[str, float]:
    return {spec["name"]: float(spec["default"]) for spec in specs}


def _iter_unet_field_specs():
    for section in UNET_SECTION_SPECS:
        for field in section["fields"]:
            yield field


def get_clip_default_config() -> dict[str, float]:
    return _collect_defaults(CLIP_FIELD_SPECS)


def get_unet_default_config() -> dict[str, float]:
    defaults = {"base_scale": 1.0}
    defaults.update(_collect_defaults(list(_iter_unet_field_specs())))
    return defaults


def normalize_vectors_override(raw_override: Any) -> list[float] | None:
    if raw_override in (None, ""):
        return None

    if isinstance(raw_override, str):
        values = [item.strip() for item in raw_override.split(",") if item.strip()]
    elif isinstance(raw_override, list):
        values = raw_override
    else:
        raise ValueError(
            "vectors_override must be a comma-separated string or a list of floats."
        )

    parsed = [float(value) for value in values]
    if len(parsed) != 19:
        raise ValueError(f"vectors_override expects 19 values, found {len(parsed)}.")
    return parsed


def build_clip_config(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    config = get_clip_default_config()
    if not overrides:
        return config

    for spec in CLIP_FIELD_SPECS:
        name = spec["name"]
        if name in overrides and overrides[name] is not None:
            config[name] = float(overrides[name])
    return config


def build_unet_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = get_unet_default_config()
    if overrides:
        for key, value in overrides.items():
            if key == "vectors_override":
                continue
            if value is not None and key in config:
                config[key] = float(value)

        vectors_override = normalize_vectors_override(overrides.get("vectors_override"))
        if vectors_override is not None:
            config["vectors_override"] = vectors_override

    return config


def build_arthemy_tuner_payload(
    mode: str = "Soft Value",
    clip_overrides: dict[str, Any] | None = None,
    unet_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in ARTHEMY_TUNER_MODES:
        raise ValueError(f"mode must be one of {ARTHEMY_TUNER_MODES}, got {mode}")

    return {
        "mode": mode,
        "clip": build_clip_config(clip_overrides),
        "unet": build_unet_config(unet_overrides),
    }


def build_arthemy_tune_job_config(
    target_model: str,
    mode: str = "Soft Value",
    clip_overrides: dict[str, Any] | None = None,
    unet_overrides: dict[str, Any] | None = None,
    output_name: str | None = None,
    save_model: bool = True,
) -> dict[str, Any]:
    target_model = str(target_model).strip()
    if not target_model:
        raise ValueError("target_model is required for Arthemy tuning.")

    config = {
        "target_model": target_model,
        "models": [],
        "save_model": bool(save_model),
        "arthemy_tuner": build_arthemy_tuner_payload(
            mode=mode,
            clip_overrides=clip_overrides,
            unet_overrides=unet_overrides,
        ),
    }
    if output_name:
        config["output_name"] = output_name
    return config
