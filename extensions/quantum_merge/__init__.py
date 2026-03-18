import os
import gc
import json
import logging
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
from module.extension_manager import (
    register_strategy,
    register_pre_config_hook,
    register_post_merge_hook,
)

# --- QuantumMerge Globals ---
_QUANTUM_HYPER_OUT = None
_QUANTUM_ENTANGLEMENT = 0.7714
_QUANTUM_CHUNK_SIZE = 2048
_QUANTUM_PROMPT = ""
_ADD_VPRED = False


def process_fft_chunked(
    param1_half: Tensor,
    param2_half: Tensor,
    hyper_out: Tensor,
    decoherence_mask: Tensor,
    chunk_size=32,
) -> Tensor:
    """QuantumMerge の FFTベースのブレンド処理"""
    device = param1_half.device
    orig_shape = param1_half.shape
    flat_shape = (-1, orig_shape[-1])

    flat1 = param1_half.view(flat_shape)
    flat2 = param2_half.view(flat_shape)
    flat_mask = decoherence_mask.view(flat_shape)

    processed_chunks = []

    for i in range(0, flat1.shape[0], chunk_size):
        with torch.no_grad():
            chunk1 = flat1[i : i + chunk_size].float()
            chunk2 = flat2[i : i + chunk_size].float()
            mask_chunk = flat_mask[i : i + chunk_size].to(device, non_blocking=True)

            fft1 = torch.fft.rfft(chunk1, dim=-1)
            fft2 = torch.fft.rfft(chunk2, dim=-1)
            freq_dim = fft1.shape[-1]

            # hyper_out のサイズ調整
            if hyper_out.shape[-1] < freq_dim:
                coeff = hyper_out.repeat(1, freq_dim // hyper_out.shape[-1] + 1)[:, :freq_dim]
            else:
                coeff = hyper_out[:, :freq_dim]

            coeff = coeff.expand(chunk1.size(0), -1).float().to(device)

            magnitude_blend = torch.sigmoid(coeff * 5)
            phase_blend = torch.sigmoid(coeff * 3 - 1)

            blended_fft_real = magnitude_blend * fft1.real + (1 - magnitude_blend) * fft2.real
            blended_fft_imag = phase_blend * fft1.imag + (1 - phase_blend) * fft2.imag
            blended_fft = torch.complex(blended_fft_real, blended_fft_imag)

            blended_chunk = torch.fft.irfft(blended_fft, n=chunk1.shape[-1], dim=-1)
            avg = (chunk1 + chunk2) / 2

            # デコヒーレンスマスク適用（一部を単なる平均に戻す）
            blended_chunk[mask_chunk] = avg[mask_chunk]

            blended_chunk = blended_chunk.to(param1_half.dtype).cpu()
            processed_chunks.append(blended_chunk)

    blended_flat = torch.cat(processed_chunks, dim=0)
    return blended_flat.view(orig_shape).to(device)


@merge_method
def strategy_quantum(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    velocity: Parameter(Tensor) = 1.0,  # 使用しないがインターフェースとして維持
    key_patterns_json: Parameter(str) = "[]",
    **kwargs,
) -> Return(Tensor):
    """
    QuantumMerge 戦略。
    プロンプトから生成された hyper_out を用いて FFT 領域でブレンドを行う。
    """
    key = kwargs.get("key", "")
    global _QUANTUM_HYPER_OUT, _QUANTUM_ENTANGLEMENT, _QUANTUM_CHUNK_SIZE, _QUANTUM_PROMPT

    # hyper_out が未生成、または weight 以外のパラメータ（bias等）は単純平均
    if _QUANTUM_HYPER_OUT is None or "weight" not in key:
        return (a + b) / 2.0

    # 乱数シードの固定（キーとプロンプトに依存した決定的なマスク生成のため）
    seed = abs(hash(_QUANTUM_PROMPT + key)) % (2**32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    # 20% の確率でデコヒーレンス（FFTブレンドを行わず平均値を使う）
    decoherence_mask = torch.rand(a.shape, generator=generator, device="cpu") < 0.2

    # FFTベースのブレンド
    blended = process_fft_chunked(a, b, _QUANTUM_HYPER_OUT, decoherence_mask, _QUANTUM_CHUNK_SIZE)

    # 量子もつれ (Entanglement) 比率での最終合成
    merged = (
        blended.float() * _QUANTUM_ENTANGLEMENT
        + (a.float() * (1 - _QUANTUM_ENTANGLEMENT) + b.float() * (1 - _QUANTUM_ENTANGLEMENT)) / 2.0
    ).to(a.dtype)

    return merged


class QuantumCLIPExtractor:
    """SDXLのチェックポイントからCLIP部分を抽出し、HuggingFace形式に変換する"""

    @classmethod
    def extract_from_checkpoint(cls, checkpoint_path: str) -> tuple[dict, dict]:
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
        components = {"clip_g": {}, "clip_l": {}}

        for key in state_dict:
            clean_key = key.replace("conditioner.embedders.0.", "").replace("cond_stage_model.", "")
            if "text_model.encoder.layers.23" in clean_key or "text_projection" in clean_key:
                components["clip_g"][clean_key] = state_dict[key]
            elif "text_model.encoder.layers" in clean_key:
                components["clip_l"][clean_key] = state_dict[key]

        return (
            cls.process_component(components["clip_g"]),
            cls.process_component(components["clip_l"]),
        )

    @staticmethod
    def process_component(component: dict) -> dict:
        processed = {}
        replacements = {
            "layer_norm1": "self_attn_layer_norm",
            "layer_norm2": "final_layer_norm",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
            "positional_embedding": "embeddings.position_embedding.weight",
            "token_embedding": "embeddings.token_embedding.weight",
        }

        for key in component:
            new_key = key
            for old, new in replacements.items():
                new_key = new_key.replace(old, new)
            processed[new_key] = component[key]
        return processed


def load_custom_clip(ckpt_path: str):
    from transformers import CLIPTextModel, CLIPTextConfig

    clip_g, clip_l = QuantumCLIPExtractor.extract_from_checkpoint(ckpt_path)
    merged_state = {**clip_g, **clip_l}

    # SDXL等は openai/clip-vit-large-patch14 ベース
    config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel(config)

    model_state = text_encoder.state_dict()
    filtered = {k: v for k, v in merged_state.items() if k in model_state}
    model_state.update(filtered)
    text_encoder.load_state_dict(model_state, strict=False)

    # 処理デバイスを決定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return text_encoder.eval().to(device)


def pre_config_quantum_setup(config: dict) -> dict:
    """
    YAML設定から QuantumMerge の初期化処理を行う。
    プロンプトから Hypernetwork を通して特徴量を事前計算しキャッシュする。
    """
    quantum_config = config.get("quantum_merge")
    if not quantum_config:
        return config

    global \
        _QUANTUM_HYPER_OUT, \
        _QUANTUM_ENTANGLEMENT, \
        _QUANTUM_CHUNK_SIZE, \
        _QUANTUM_PROMPT, \
        _ADD_VPRED

    _QUANTUM_PROMPT = quantum_config.get("prompt", "")
    _QUANTUM_ENTANGLEMENT = quantum_config.get("entanglement", 0.7714)
    _QUANTUM_CHUNK_SIZE = quantum_config.get("chunk_size", 2048)
    _ADD_VPRED = quantum_config.get("add_vpred", False)
    clip_source_path = quantum_config.get("clip_source_model")

    if not _QUANTUM_PROMPT or not clip_source_path:
        logging.warning(
            "QuantumMerge: 'prompt' または 'clip_source_model' が指定されていません。処理をスキップします。"
        )
        return config

    if not os.path.exists(clip_source_path):
        logging.warning(f"QuantumMerge: CLIP抽出用モデルが見つかりません: {clip_source_path}")
        return config

    try:
        from transformers import CLIPTokenizer
        import torch.nn as nn

        logging.info("QuantumMerge: CLIPモデルを抽出・ロードし、プロンプト特徴量を計算します...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_encoder = load_custom_clip(clip_source_path)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        hypernet = nn.Sequential(
            nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 256), nn.Tanh()
        ).to(device)

        # dtype調整 (FP16が使える環境なら合わせる)
        if device == "cuda":
            hypernet = hypernet.half()
            text_encoder = text_encoder.half()

        with torch.no_grad():
            text_inputs = tokenizer(
                _QUANTUM_PROMPT,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            text_emb = text_encoder(text_input_ids).pooler_output

            if device == "cuda":
                text_emb = text_emb.half()

            _QUANTUM_HYPER_OUT = hypernet(text_emb).float().cpu()  # メモリ節約のためCPU退避

        logging.info("QuantumMerge: 特徴量の計算が完了しました。")

        # クリーンアップ (VRAM解放)
        del text_encoder
        del hypernet
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"QuantumMerge 初期化エラー: {e}")

    return config


def post_merge_add_vpred(config: dict, output_path: str):
    """
    出力モデルに強制的に空の v_pred テンソルを追加し、
    ファイル名に '_s' を付与する処理。
    """
    global _ADD_VPRED
    if not _ADD_VPRED:
        return

    if not output_path.endswith(".safetensors"):
        return

    logging.info("QuantumMerge: 出力モデルに v_pred テンソルを追加します...")
    try:
        from safetensors.torch import save_file
        from safetensors import safe_open

        # Use lazy loading with safe_open and yield tensors one by one
        # to prevent loading the entire model into RAM at once
        vpred_path = output_path.replace(".safetensors", "_s.safetensors")

        with safe_open(output_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}

            # create a generator/dictionary-like object that yields tensors
            class LazyDict(dict):
                def keys(self) -> list:  # type: ignore
                    return list(f.keys()) + ["v_pred"]

                def items(self):  # type: ignore
                    for k in f.keys():
                        yield k, f.get_tensor(k)
                    yield "v_pred", torch.tensor([])

                def __getitem__(self, key):
                    if key == "v_pred":
                        return torch.tensor([])
                    return f.get_tensor(key)

                def __iter__(self):
                    return iter(self.keys())

                def __len__(self):
                    return len(f.keys()) + 1

            lazy_tensors = LazyDict()
            save_file(lazy_tensors, vpred_path, metadata=metadata)

        # 元のファイルを削除してリネーム（置換）するか、別ファイルとして残すか
        # 元実装に合わせ別名保存後、元の出力ファイルを削除する
        os.remove(output_path)
        os.rename(vpred_path, output_path)

        logging.info("v_pred の追加が完了しました。")
    except Exception as e:
        logging.error(f"v_pred の追加に失敗しました: {e}")


def setup():
    register_pre_config_hook(pre_config_quantum_setup)
    register_strategy("quantum", strategy_quantum)
    register_post_merge_hook(post_merge_add_vpred)
