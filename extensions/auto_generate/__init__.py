import os
import logging

from module.extension_manager import register_post_merge_hook
from module.history import update_history_entry
from module.services.generation import (
    GenerationRequest,
    build_history_image_update,
    generate_and_collect_artifacts,
)

logger = logging.getLogger(__name__)

# グローバルな自動生成の設定を保持する
# TODO: 今後、UIからこの設定を上書きできるような仕組みを用意する
auto_generate_config = {
    "enabled": False,
    "prompt": "1girl, masterpiece, best quality, highly detailed",
    "negative_prompt": "worst quality, low quality, blurry",
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg": 7.0,
    "sampler_name": "euler",
    "scheduler": "normal",
    "seed": -1,
}


def auto_generate_hook(config_dict: dict, output_path: str):
    """マージ完了後に呼び出されるフック"""

    # 拡張機能の設定で有効になっていなければ何もしない
    if not auto_generate_config.get("enabled", False):
        return

    logger.info(f"自動生成を開始します。対象モデル: {output_path}")

    # パラメータの準備
    prompt = auto_generate_config.get("prompt", "")
    neg_prompt = auto_generate_config.get("negative_prompt", "")
    w = auto_generate_config.get("width", 512)
    h = auto_generate_config.get("height", 512)
    steps = auto_generate_config.get("steps", 20)
    cfg = auto_generate_config.get("cfg", 7.0)
    sampler_name = auto_generate_config.get("sampler_name", "euler")
    scheduler = auto_generate_config.get("scheduler", "normal")
    seed = auto_generate_config.get("seed", -1)

    result = generate_and_collect_artifacts(
        GenerationRequest(
            model_path=output_path,
            output_dir=os.path.dirname(output_path),
            output_prefix=f"{os.path.splitext(os.path.basename(output_path))[0]}_sample",
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            seed=seed,
        )
    )

    if not result.artifacts:
        logger.error("自動生成に失敗しました。")
        return

    for artifact in result.artifacts:
        logger.info(f"自動生成画像を保存しました: {artifact.path}")

    # 最初の生成画像をヒストリのプレビュー用に登録する
    update_dict = build_history_image_update(result)
    if update_dict:
        success = update_history_entry(output_path, update_dict)
        if success:
            logger.info(f"ヒストリエントリの画像を更新しました: {output_path}")
        else:
            logger.warning(f"ヒストリエントリが見つからなかったため画像パスの更新をスキップしました: {output_path}")


def setup():
    """拡張機能の初期化処理"""
    register_post_merge_hook(auto_generate_hook)
    logger.info("Auto Generate Extension initialized.")
