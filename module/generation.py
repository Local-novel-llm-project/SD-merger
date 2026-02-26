import os
import sys
import torch
import logging


def _get_comfy_dir():
    """Returns the path to the ComfyUI reference directory."""
    return os.environ.get(
        "COMFYUI_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference", "ComfyUI"))
    )


def generate_image(
    model_path,
    prompt="A beautiful landscape, high quality, detailed",
    negative_prompt="blurry, bad quality, low res",
    width=512,
    height=512,
    steps=20,
    cfg=8.0,
    sampler_name="euler",
    scheduler="normal",
    seed=1337,
):
    """
    Generates an image using the ComfyUI backend.
    """
    comfy_dir = _get_comfy_dir()
    if not os.path.exists(comfy_dir):
        logging.error(f"ComfyUI directory not found at {comfy_dir}. Please install it to use generation testing.")
        return None

    if comfy_dir not in sys.path:
        sys.path.append(comfy_dir)

    try:
        import folder_paths
        import nodes
    except ImportError as e:
        logging.error(f"Failed to import ComfyUI modules from {comfy_dir}: {e}")
        return None

    try:
        # Avoid model not found errors by temporarily adding the directory to ComfyUI's search path
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)

        # Add to custom paths safely
        if model_dir not in folder_paths.get_folder_paths("checkpoints"):
            folder_paths.add_model_folder_path("checkpoints", model_dir)

        print(f"Loading checkpoint: {model_name}")
        ckpt_loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = ckpt_loader.load_checkpoint(ckpt_name=model_name)

        print("Encoding prompts...")
        clip_pos = nodes.CLIPTextEncode().encode(clip=clip, text=prompt)[0]
        clip_neg = nodes.CLIPTextEncode().encode(clip=clip, text=negative_prompt)[0]

        print(f"Generating empty latent ({width}x{height})...")
        latent = nodes.EmptyLatentImage().generate(width=width, height=height, batch_size=1)[0]

        print("Sampling...")
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

        print("Decoding VAE...")
        images = nodes.VAEDecode().decode(samples=samples, vae=vae)[0]

        # Convert tensor to PIL Image
        # The tensor is (batch, h, w, c) in range [0, 1]
        images = torch.clamp(images, 0.0, 1.0)
        images = (images * 255.0).to(torch.uint8).cpu().numpy()

        from PIL import Image

        pil_images = [Image.fromarray(img) for img in images]

        # Memory cleanup
        try:
            import comfy.model_management as mm
            del model, clip, vae, latent, samples, images
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pil_images[0]  # Return the first image

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback

        traceback.print_exc()
        return None
