import os
import logging
import torch
from module.extension_manager import register_post_merge_hook


def _load_existing_metadata(output_path: str) -> dict[str, str]:
    import safetensors

    existing_metadata: dict[str, str] = {}
    with safetensors.safe_open(output_path, framework="pt", device="cpu") as f:
        meta = f.metadata()
        if meta:
            existing_metadata.update(meta)
    return existing_metadata


def _load_vae_state_dict(bake_in_vae: str):
    from safetensors.torch import load_file

    if bake_in_vae.endswith(".safetensors"):
        return load_file(bake_in_vae, device="cpu")

    vae_dict = torch.load(
        bake_in_vae, map_location="cpu", weights_only=True
    )
    if "state_dict" in vae_dict:
        return vae_dict["state_dict"]
    return vae_dict


def _merge_vae_into_state_dict(state_dict, bake_in_vae: str) -> None:
    if not bake_in_vae or not os.path.exists(bake_in_vae):
        return

    logging.info(f"VAE ({bake_in_vae}) を出力モデルに焼き込んでいます...")
    vae_dict = _load_vae_state_dict(bake_in_vae)

    for key in vae_dict.keys():
        theta_0_key = "first_stage_model." + key
        if theta_0_key in state_dict:
            state_dict[theta_0_key] = vae_dict[key].to(
                state_dict[theta_0_key].dtype
            )
        elif key in state_dict:
            state_dict[key] = vae_dict[key].to(state_dict[key].dtype)


def _merge_custom_metadata(existing_metadata: dict[str, str], metadata_config: dict) -> None:
    if not metadata_config:
        return

    logging.info("カスタムメタデータを追記しています...")
    for k, v in metadata_config.items():
        existing_metadata[str(k)] = str(v)


def post_merge_utils(config: dict, output_path: str):
    """
    出力された .safetensors ファイルに:
    1. VAE を焼き込む (bake_in_vae)
    2. カスタムメタデータを追記・上書きする (custom_metadata)
    """
    bake_in_vae = config.get("bake_in_vae", None)
    metadata_config = config.get("custom_metadata", None)

    if not bake_in_vae and not metadata_config:
        return

    if not output_path.endswith(".safetensors"):
        logging.warning("これらの機能は .safetensors 出力時のみサポートされます。")
        return

    from safetensors.torch import load_file, save_file

    logging.info(
        "出力モデルの事後処理 (VAE焼き込み/メタデータ付与) を開始します..."
    )
    state_dict = load_file(output_path, device="cpu")
    existing_metadata = _load_existing_metadata(output_path)

    _merge_vae_into_state_dict(state_dict, bake_in_vae)
    _merge_custom_metadata(existing_metadata, metadata_config)

    temp_path = output_path + ".tmp"
    try:
        save_file(state_dict, temp_path, metadata=existing_metadata)
        os.replace(temp_path, output_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    logging.info("事後処理が完了しました。")


def setup():
    register_post_merge_hook(post_merge_utils)
