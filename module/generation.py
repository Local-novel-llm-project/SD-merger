"""画像生成モジュール (diffusers バックエンド)

safetensors チェックポイントを diffusers の from_single_file() で読み込み、
ComfyUI を必要とせずに画像生成を実行する。
"""

import gc
import logging
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ComfyUI 形式のサンプラー名 → diffusers スケジューラクラス名の対応表
_SCHEDULER_MAP: dict[str, str] = {
    "euler": "EulerDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
    "dpm_2": "KDPM2DiscreteScheduler",
    "dpm_2_ancestral": "KDPM2AncestralDiscreteScheduler",
    "lms": "LMSDiscreteScheduler",
    "dpm_fast": "DPMSolverMultistepScheduler",
    "dpm_adaptive": "DPMSolverMultistepScheduler",
    "dpmpp_2s_ancestral": "DPMSolverSinglestepScheduler",
    "dpmpp_sde": "DPMSolverSDEScheduler",
    "dpmpp_2m": "DPMSolverMultistepScheduler",
    "ddim": "DDIMScheduler",
    "uni_pc": "UniPCMultistepScheduler",
    "uni_pc_bh2": "UniPCMultistepScheduler",
}


def _is_sdxl_checkpoint(model_path: str) -> bool:
    """safetensors ファイルのキー構造から SDXL かどうかを判定する。

    ファイル全体を読み込まず、メタデータ（キー名一覧）だけを参照するので高速。
    """
    try:
        from safetensors import safe_open

        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            # SDXL 特有のキーをチェック
            for key in keys:
                if "conditioner.embedders.1" in key or "input_blocks.7.1" in key:
                    return True
            return False
    except Exception:
        # safetensors の読み込みに失敗した場合はファイルサイズで推定
        try:
            size_gb = Path(model_path).stat().st_size / (1024**3)
            return size_gb > 5.0  # SDXL は通常 6GB 以上
        except Exception:
            return False


def _get_scheduler(sampler_name: str, scheduler_type: str, current_config: dict):
    """サンプラー名とスケジューラタイプからスケジューラインスタンスを生成する。"""
    import diffusers

    scheduler_cls_name = _SCHEDULER_MAP.get(sampler_name, "EulerDiscreteScheduler")
    scheduler_cls = getattr(diffusers, scheduler_cls_name, None)
    if scheduler_cls is None:
        logger.warning(f"スケジューラ '{scheduler_cls_name}' が見つかりません。EulerDiscreteScheduler を使用します。")
        from diffusers import EulerDiscreteScheduler

        scheduler_cls = EulerDiscreteScheduler

    # karras シグマの適用
    kwargs = {}
    if scheduler_type == "karras" and hasattr(scheduler_cls, "use_karras_sigmas"):
        kwargs["use_karras_sigmas"] = True

    try:
        return scheduler_cls.from_config(current_config, **kwargs)
    except Exception as e:
        logger.warning(f"スケジューラ設定の適用に失敗: {e}。デフォルト設定を使用します。")
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(current_config)


def generate_image(
    model_path: str,
    prompt: str = "A beautiful landscape, high quality, detailed",
    negative_prompt: str = "blurry, bad quality, low res",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 8.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    seed: int = 1337,
) -> Image.Image | None:
    """diffusers を使って画像を生成する。

    Args:
        model_path: safetensors チェックポイントへのパス。
        prompt: ポジティブプロンプト。
        negative_prompt: ネガティブプロンプト。
        width: 画像の幅。
        height: 画像の高さ。
        steps: サンプリングステップ数。
        cfg: CFG スケール。
        sampler_name: サンプラー名 (ComfyUI 形式の名前をサポート)。
        scheduler: スケジューラタイプ ("normal", "karras" 等)。
        seed: 乱数シード。

    Returns:
        生成された PIL Image、失敗した場合は None。
    """
    if not Path(model_path).exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        return None

    try:
        # SDXL かどうかを判定してパイプラインを選択
        is_sdxl = _is_sdxl_checkpoint(model_path)

        if is_sdxl:
            from diffusers import StableDiffusionXLPipeline

            logger.info(f"SDXL チェックポイントを読み込み中: {model_path}")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        else:
            from diffusers import StableDiffusionPipeline

            logger.info(f"SD1.5 チェックポイントを読み込み中: {model_path}")
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

        # デバイスの決定
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            logger.warning("CUDA が利用できません。CPU で実行します（低速）。")

        pipe = pipe.to(device)

        # メモリ最適化
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        # スケジューラの設定
        pipe.scheduler = _get_scheduler(sampler_name, scheduler, pipe.scheduler.config)

        # 生成
        generator = torch.Generator(device=device).manual_seed(seed)

        logger.info(f"画像を生成中... (steps={steps}, cfg={cfg}, " f"size={width}x{height}, seed={seed})")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )

        image = result.images[0]

        # メモリ解放
        del pipe, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("画像生成が完了しました。")
        return image

    except Exception as e:
        logger.error(f"画像生成中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()
        return None
