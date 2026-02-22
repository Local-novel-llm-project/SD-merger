import os
import logging
import torch
from module.extension_manager import register_post_merge_hook


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

    try:
        from safetensors.torch import load_file, save_file
        import safetensors

        logging.info(
            "出力モデルの事後処理 (VAE焼き込み/メタデータ付与) を開始します..."
        )
        state_dict = load_file(output_path, device="cpu")

        existing_metadata = {}
        with safetensors.safe_open(output_path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            if meta:
                existing_metadata.update(meta)

        # 1. Bake in VAE
        if bake_in_vae and os.path.exists(bake_in_vae):
            logging.info(f"VAE ({bake_in_vae}) を出力モデルに焼き込んでいます...")
            if bake_in_vae.endswith(".safetensors"):
                vae_dict = load_file(bake_in_vae, device="cpu")
            else:
                vae_dict = torch.load(
                    bake_in_vae, map_location="cpu", weights_only=True
                )
                if "state_dict" in vae_dict:
                    vae_dict = vae_dict["state_dict"]

            for key in vae_dict.keys():
                theta_0_key = "first_stage_model." + key
                # SD 系の VAE ならそのままコピー（SDXLでも同様）
                if theta_0_key in state_dict:
                    state_dict[theta_0_key] = vae_dict[key].to(
                        state_dict[theta_0_key].dtype
                    )
                elif key in state_dict:
                    # SDXL などプレフィックスがない VAE の場合
                    state_dict[key] = vae_dict[key].to(state_dict[key].dtype)

        # 2. Custom Metadata
        if metadata_config:
            logging.info("カスタムメタデータを追記しています...")
            for k, v in metadata_config.items():
                existing_metadata[str(k)] = str(v)

        temp_path = output_path + ".tmp"
        save_file(state_dict, temp_path, metadata=existing_metadata)

        os.replace(temp_path, output_path)
        logging.info("事後処理が完了しました。")

    except Exception as e:
        logging.error(f"事後処理に失敗しました: {e}")


def setup():
    register_post_merge_hook(post_merge_utils)
