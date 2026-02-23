import torch
import safetensors.torch
import matplotlib.pyplot as plt
import io
from PIL import Image
import re


def categorize_key(key):
    """Categorize standard SD unet keys into IN, MID, OUT blocks"""
    # SD 1.5 standard / diffusers standard
    match_in = re.search(r"input_blocks\.(\d+)", key) or re.search(
        r"down_blocks\.(\d+)", key
    )
    if match_in:
        return f"IN{int(match_in.group(1)):02d}"

    match_mid = re.search(r"middle_block", key) or re.search(r"mid_block", key)
    if match_mid:
        return "MID00"

    match_out = re.search(r"output_blocks\.(\d+)", key) or re.search(
        r"up_blocks\.(\d+)", key
    )
    if match_out:
        return f"OUT{int(match_out.group(1)):02d}"

    if "time_embed" in key:
        return "TIME"

    if "label_emb" in key or "class_emb" in key:
        return "COND"

    return "OTHER"


def analyze_models(model_a_path, model_b_path, metric="Cosine Similarity"):
    """
    Compare two safetensors models block by block and return a matplotlib figure.
    """
    print(f"Loading {model_a_path}...")
    sd_a = safetensors.torch.load_file(model_a_path)
    print(f"Loading {model_b_path}...")
    sd_b = safetensors.torch.load_file(model_b_path)

    # Identify common keys
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    common_keys = keys_a.intersection(keys_b)

    if not common_keys:
        raise ValueError("No common keys found between the two models.")

    # Only process UNet or Transformer keys
    unet_keys = [
        k for k in common_keys if "diffusion_model" in k or "unet" in k or "model." in k
    ]
    if not unet_keys:
        unet_keys = list(common_keys)  # Fallback to all keys if no unet detected

    blocks_data = {}

    for k in unet_keys:
        block_name = categorize_key(k)

        t_a = sd_a[k].float()
        t_b = sd_b[k].float()

        if t_a.shape != t_b.shape:
            continue

        t_a_flat = t_a.view(-1)
        t_b_flat = t_b.view(-1)

        # Calculate metric
        if metric == "Cosine Similarity":
            # Avoid division by zero
            norm_a = t_a_flat.norm()
            norm_b = t_b_flat.norm()
            if norm_a == 0 or norm_b == 0:
                val = 1.0 if norm_a == norm_b else 0.0
            else:
                val = torch.nn.functional.cosine_similarity(
                    t_a_flat.unsqueeze(0), t_b_flat.unsqueeze(0)
                ).item()
        elif metric == "Euclidean Distance":
            val = torch.dist(t_a_flat, t_b_flat, p=2).item()
        elif metric == "Mean Absolute Difference":
            val = torch.mean(torch.abs(t_a_flat - t_b_flat)).item()
        else:
            val = 0.0

        if block_name not in blocks_data:
            blocks_data[block_name] = []
        blocks_data[block_name].append(val)

    # Aggregate data
    aggregated = {}
    for block, values in blocks_data.items():
        if values:
            aggregated[block] = sum(values) / len(values)

    # Sort blocks
    def block_sort_key(k):
        if k.startswith("IN"):
            return (0, k)
        if k.startswith("MID"):
            return (1, k)
        if k.startswith("OUT"):
            return (2, k)
        if k == "TIME":
            return (3, k)
        return (4, k)

    sorted_blocks = sorted(aggregated.keys(), key=block_sort_key)
    sorted_values = [aggregated[b] for b in sorted_blocks]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = []
    for b in sorted_blocks:
        if b.startswith("IN"):
            colors.append("#3498db")
        elif b.startswith("MID"):
            colors.append("#f1c40f")
        elif b.startswith("OUT"):
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")

    bars = ax.bar(sorted_blocks, sorted_values, color=colors)

    ax.set_title(
        f"Model Difference Analysis ({metric})\n{model_a_path.split('/')[-1]} vs {model_b_path.split('/')[-1]}"
    )
    ax.set_ylabel(metric)
    ax.set_xlabel("UNet Blocks")
    plt.xticks(rotation=45, ha="right")

    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    return fig
