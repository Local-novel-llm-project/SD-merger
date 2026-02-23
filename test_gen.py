import os
import sys
import torch

comfy_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "refrence", "ComfyUI")
)
sys.path.append(comfy_dir)

import folder_paths
import nodes


def generate_image(
    model_path,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    sampler_name,
    scheduler,
    seed,
):
    # Set custom paths so ComfyUI doesn't clutter
    folder_paths.add_model_folder_path("checkpoints", os.path.dirname(model_path))
    model_name = os.path.basename(model_path)

    # 1. Load Checkpoint
    ckpt_loader = nodes.CheckpointLoaderSimple()
    try:
        model, clip, vae = ckpt_loader.load_checkpoint(ckpt_name=model_name)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    # 2. CLIP Text Encode
    clip_pos = nodes.CLIPTextEncode().encode(clip=clip, text=prompt)[0]
    clip_neg = nodes.CLIPTextEncode().encode(clip=clip, text=negative_prompt)[0]

    # 3. Empty Latent
    latent = nodes.EmptyLatentImage().generate(
        width=width, height=height, batch_size=1
    )[0]

    # 4. KSampler
    ksampler = nodes.KSampler()
    samples = ksampler.sample(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=clip_pos,
        negative=clip_neg,
        latent_image=latent,
        denoise=1.0,
    )[0]

    # 5. VAE Decode
    images = nodes.VAEDecode().decode(samples=samples, vae=vae)[0]

    # images is a torch.Tensor of shape (batch, height, width, channels)
    return images


if __name__ == "__main__":
    # Just a compilation test
    print("Test compilation success")
