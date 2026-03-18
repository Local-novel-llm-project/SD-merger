import os
import time
import logging
from PIL import Image

from module.extension_manager import register_post_merge_hook
from module.generation import generate_image
from module.history import update_history_entry

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

    import random

    actual_seed = seed if seed > 0 else random.randint(1, 1125899906842624)

    # 複数枚生成をサポートしているため画像のリストが返る
    images = generate_image(
        model_path=output_path,
        prompt=prompt,
        negative_prompt=neg_prompt,
        width=w,
        height=h,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        seed=actual_seed,
    )

    if not images:
        logger.error("自動生成に失敗しました。")
        return

    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    generated_image_paths = []

    # 画像の保存
    for i, img in enumerate(images):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"{base_name}_sample_{timestamp}_{i}.png"
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path)
        logger.info(f"自動生成画像を保存しました: {img_path}")
        generated_image_paths.append(img_path)

    # 最初の生成画像をヒストリのプレビュー用に登録する
    if generated_image_paths:
        update_dict = {"preview_image": generated_image_paths[0], "generated_images": generated_image_paths}
        success = update_history_entry(output_path, update_dict)
        if success:
            logger.info(f"ヒストリエントリの画像を更新しました: {output_path}")
        else:
            logger.warning(f"ヒストリエントリが見つからなかったため画像パスの更新をスキップしました: {output_path}")


def setup():
    """拡張機能の初期化処理"""
    register_post_merge_hook(auto_generate_hook)
    logger.info("Auto Generate Extension initialized.")
