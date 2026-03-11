import logging

from module.extension_manager import register_pre_config_hook
from module.pipeline.poison import run_poison_merge

logger = logging.getLogger(__name__)


def run_poison_pipeline(config: dict) -> dict:
    """
    設定ファイルの poison_merge セクションを処理するフック。
    反復的に LoRA をモデルにマージし、各ステップで画像生成を行う。
    """
    poison_config = config.get("poison_merge")
    if not poison_config:
        return config

    logger.info("Poison Merge パイプラインを開始します...")
    last_output_path = run_poison_merge(config)

    logger.info("Poison Merge パイプラインが完了しました。")

    config["_skip_merge"] = True
    config["_skip_merge_output"] = last_output_path or poison_config.get("base_model")

    return config


def setup():
    register_pre_config_hook(run_poison_pipeline)
