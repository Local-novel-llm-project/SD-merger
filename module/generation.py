"""画像生成モジュール (diffusers バックエンド)

safetensors チェックポイントを diffusers の from_single_file() で読み込み、
ComfyUI を必要とせずに画像生成を実行する。
"""

import collections
import gc
from pathlib import Path

import torch
from PIL import Image

from module.logging_config import logger
from module.exceptions import GenerationError

# --- モデルキャッシュ管理設定 ---
MAX_CACHE_SIZE_GB = 10.0  # キャッシュするモデルの最大サイズ(GB)
# _MODEL_CACHE は OrderedDict を用いて LRU (Least Recently Used) キャッシュを実装する
# 構造: { model_path: {"pipe": StableDiffusionPipeline, "size_gb": float} }
_MODEL_CACHE: collections.OrderedDict = collections.OrderedDict()

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
        logger.warning(
            f"スケジューラ '{scheduler_cls_name}' が見つかりません。EulerDiscreteScheduler を使用します。"
        )
        from diffusers import EulerDiscreteScheduler

        scheduler_cls = EulerDiscreteScheduler

    # karras シグマの適用
    kwargs = {}
    if scheduler_type == "karras" and hasattr(scheduler_cls, "use_karras_sigmas"):
        kwargs["use_karras_sigmas"] = True

    try:
        return scheduler_cls.from_config(current_config, **kwargs)
    except Exception as e:
        logger.warning(
            f"スケジューラ設定の適用に失敗: {e}。デフォルト設定を使用します。"
        )
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(current_config)


def clear_model_cache():
    """現在のモデルキャッシュをすべてクリアする"""
    global _MODEL_CACHE
    cleared_count = len(_MODEL_CACHE)

    for _, cache_data in _MODEL_CACHE.items():
        pipe = cache_data.get("pipe")
        if pipe is not None:
            del pipe

    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"モデルキャッシュをクリアしました。({cleared_count}件)")
    return cleared_count


def get_cache_info() -> dict:
    """現在のキャッシュの状態(読み込み済みのモデルとそのサイズ)を返す"""
    total_gb = sum(item["size_gb"] for item in _MODEL_CACHE.values())
    models = list(_MODEL_CACHE.keys())
    return {
        "total_gb": total_gb,
        "max_gb": MAX_CACHE_SIZE_GB,
        "models": models,
        "count": len(models),
    }


def _evict_cache_if_needed(new_size_gb: float):
    """
    新しく追加するサイズ new_size_gb を考慮して、
    合計サイズが MAX_CACHE_SIZE_GB を超える場合は古いキャッシュ(LRU)から破棄する。
    """
    global _MODEL_CACHE
    while _MODEL_CACHE:
        total_gb = sum(item["size_gb"] for item in _MODEL_CACHE.values())
        if total_gb + new_size_gb <= MAX_CACHE_SIZE_GB:
            break

        # サイズ上限を超える場合は一番古いもの（先頭）を pop
        lru_path, cache_data = _MODEL_CACHE.popitem(last=False)
        pipe = cache_data.get("pipe")
        logger.info(
            f"キャッシュの容量上限に達しました。古いモデルを解放します: {lru_path} ({cache_data['size_gb']:.2f} GB)"
        )
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_cached_pipeline(model_path: str):
    """LRUキャッシュからパイプラインを取得する。無ければ読み込む。"""
    global _MODEL_CACHE

    if model_path in _MODEL_CACHE:
        # キャッシュヒット: 要素を末尾（最新）に移動する
        logger.info(f"キャッシュからモデルを取得しました: {model_path}")
        _MODEL_CACHE.move_to_end(model_path)
        return _MODEL_CACHE[model_path]["pipe"]

    # キャッシュミス: 新しく読み込む
    is_sdxl = _is_sdxl_checkpoint(model_path)

    # サイズ見積もり（とりあえずファイルサイズを使用）
    try:
        size_gb = Path(model_path).stat().st_size / (1024**3)
    except Exception:
        size_gb = 6.0 if is_sdxl else 2.0  # fallback

    _evict_cache_if_needed(size_gb)

    if is_sdxl:
        from diffusers import StableDiffusionXLPipeline

        logger.info(
            f"SDXL チェックポイントを読み込み中: {model_path} ({size_gb:.2f} GB)"
        )
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionPipeline

        logger.info(
            f"SD1.5 チェックポイントを読み込み中: {model_path} ({size_gb:.2f} GB)"
        )
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

    # デバイス・最適化の設定
    if torch.cuda.is_available():
        device = "cuda"
        pipe = pipe.to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    else:
        device = "cpu"
        logger.warning("CUDA が利用できません。CPU で実行します（低速）。")
        pipe = pipe.to(device)

    # キャッシュに保存
    _MODEL_CACHE[model_path] = {"pipe": pipe, "size_gb": size_gb}
    return pipe


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
) -> list[Image.Image] | None:
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
        生成された PIL Image のリスト（プロンプトが複数行なら複数枚）、失敗した場合は None。
    """
    if not Path(model_path).exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        return None

    try:
        pipe = get_cached_pipeline(model_path)
        device = pipe.device

        # スケジューラの設定
        pipe.scheduler = _get_scheduler(sampler_name, scheduler, pipe.scheduler.config)

        # 複数プロンプトのパース（改行区切り）
        prompts = [p.strip() for p in prompt.split("\n") if p.strip()]
        if not prompts:
            logger.warning(
                "プロンプトが空です。1枚の画像をデフォルトプロンプト生成します。"
            )
            prompts = [""]

        negative_prompts = [negative_prompt] * len(prompts)

        # 生成
        generated_images = []
        for i, (p, np) in enumerate(zip(prompts, negative_prompts)):
            # シードの決定（複数枚の場合はシードをずらす）
            current_seed = seed + i if seed > 0 else seed
            generator = torch.Generator(device=device)
            if current_seed > 0:
                generator.manual_seed(current_seed)
            else:
                generator.seed()

            logger.info(
                f"画像を生成中 [{i + 1}/{len(prompts)}]... (steps={steps}, cfg={cfg}, size={width}x{height}, seed={current_seed})"
            )
            result = pipe(
                prompt=p,
                negative_prompt=np,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
            generated_images.append(result.images[0])

        logger.info(f"画像生成が完了しました。（計 {len(generated_images)} 枚）")
        return generated_images

    except Exception as e:
        logger.error(f"画像生成中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()
        raise GenerationError(f"Image generation failed: {e}", original_error=e)


def generate_first_image(*args, **kwargs) -> Image.Image | None:
    images = generate_image(*args, **kwargs)
    if not images:
        return None
    return images[0]
