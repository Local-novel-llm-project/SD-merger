import logging
from module.extension_manager import register_pre_config_hook

# SD 1.5 / 2.X (26 elements)
_SD15_KEYWORDS = [
    "cond_stage_model",  # BASE
    # IN00 - IN11
    "model.diffusion_model.input_blocks.0.",
    "model.diffusion_model.input_blocks.1.",
    "model.diffusion_model.input_blocks.2.",
    "model.diffusion_model.input_blocks.3.",
    "model.diffusion_model.input_blocks.4.",
    "model.diffusion_model.input_blocks.5.",
    "model.diffusion_model.input_blocks.6.",
    "model.diffusion_model.input_blocks.7.",
    "model.diffusion_model.input_blocks.8.",
    "model.diffusion_model.input_blocks.9.",
    "model.diffusion_model.input_blocks.10.",
    "model.diffusion_model.input_blocks.11.",
    # MID
    "model.diffusion_model.middle_block",
    # OUT00 - OUT11
    "model.diffusion_model.output_blocks.0.",
    "model.diffusion_model.output_blocks.1.",
    "model.diffusion_model.output_blocks.2.",
    "model.diffusion_model.output_blocks.3.",
    "model.diffusion_model.output_blocks.4.",
    "model.diffusion_model.output_blocks.5.",
    "model.diffusion_model.output_blocks.6.",
    "model.diffusion_model.output_blocks.7.",
    "model.diffusion_model.output_blocks.8.",
    "model.diffusion_model.output_blocks.9.",
    "model.diffusion_model.output_blocks.10.",
    "model.diffusion_model.output_blocks.11.",
]

# SDXL (20 elements)
_SDXL_KEYWORDS = [
    "conditioner",  # BASE (SDXL uses conditioner)
    # IN00 - IN08
    "model.diffusion_model.input_blocks.0.",
    "model.diffusion_model.input_blocks.1.",
    "model.diffusion_model.input_blocks.2.",
    "model.diffusion_model.input_blocks.3.",
    "model.diffusion_model.input_blocks.4.",
    "model.diffusion_model.input_blocks.5.",
    "model.diffusion_model.input_blocks.6.",
    "model.diffusion_model.input_blocks.7.",
    "model.diffusion_model.input_blocks.8.",
    # MID
    "model.diffusion_model.middle_block",
    # OUT00 - OUT08
    "model.diffusion_model.output_blocks.0.",
    "model.diffusion_model.output_blocks.1.",
    "model.diffusion_model.output_blocks.2.",
    "model.diffusion_model.output_blocks.3.",
    "model.diffusion_model.output_blocks.4.",
    "model.diffusion_model.output_blocks.5.",
    "model.diffusion_model.output_blocks.6.",
    "model.diffusion_model.output_blocks.7.",
    "model.diffusion_model.output_blocks.8.",
]


def mbw_pre_config_hook(config: dict) -> dict:
    models = config.get("models", [])
    new_models = []

    for model_entry in models:
        mbw_str = model_entry.get("mbw")
        if not mbw_str:
            new_models.append(model_entry)
            continue

        logging.info(f"[supermerger_mbw] MBW 設定を検出しました: {mbw_str}")
        try:
            ratios = [float(r.strip()) for r in mbw_str.split(",")]
        except ValueError:
            logging.error(f"[supermerger_mbw] MBW パースエラー。数値とカンマのみを使用してください: {mbw_str}")
            new_models.append(model_entry)
            continue

        if len(ratios) == 26:
            keywords = _SD15_KEYWORDS
            logging.info("[supermerger_mbw] 26 ブロック (SD1.5/2.X) の MBW として処理します。")
        elif len(ratios) == 20:
            keywords = _SDXL_KEYWORDS
            logging.info("[supermerger_mbw] 20 ブロック (SDXL) の MBW として処理します。")
        else:
            logging.error(f"[supermerger_mbw] ブロック数が 26 または 20 ではありません (現在: {len(ratios)})")
            new_models.append(model_entry)
            continue

        # MBW指定を展開
        for ratio, key_pattern in zip(ratios, keywords):
            if ratio == 0.0:
                continue

            new_entry = dict(model_entry)
            new_entry.pop("mbw", None)
            new_entry["velocity"] = ratio
            # key_patterns がもともと設定されている場合は交差させる必要があるが、基本は上書き前提とする
            new_entry["key_patterns"] = [key_pattern]
            new_models.append(new_entry)

    config["models"] = new_models
    return config


def setup():
    logging.info("Supermerger MBW Extension の初期化を開始します。")
    register_pre_config_hook(mbw_pre_config_hook)
    logging.info("Supermerger MBW Extension の登録が完了しました。")
