import torch
import safetensors.torch
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns


def categorize_key(key):
    """Categorize standard SD unet keys into IN, MID, OUT blocks"""
    # SD 1.5 standard / diffusers standard
    match_in = re.search(r"input_blocks\.(\d+)", key) or re.search(r"down_blocks\.(\d+)", key)
    if match_in:
        return f"IN{int(match_in.group(1)):02d}"

    match_mid = re.search(r"middle_block", key) or re.search(r"mid_block", key)
    if match_mid:
        return "MID00"

    match_out = re.search(r"output_blocks\.(\d+)", key) or re.search(r"up_blocks\.(\d+)", key)
    if match_out:
        return f"OUT{int(match_out.group(1)):02d}"

    if "time_embed" in key:
        return "TIME"

    if "label_emb" in key or "class_emb" in key:
        return "COND"

    # SDXL specific conditioning keys
    if "conditioner" in key or "pooler" in key:
        return "COND_XL"

    # SDXL specific down/up blocks if not matched above
    if re.search(r"down_blocks\.(\d+)", key):
        return f"IN_XL{int(re.search(r'down_blocks\.(\d+)', key).group(1)):02d}"
    if re.search(r"up_blocks\.(\d+)", key):
        return f"OUT_XL{int(re.search(r'up_blocks\.(\d+)', key).group(1)):02d}"

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
    unet_keys = [k for k in common_keys if "diffusion_model" in k or "unet" in k or "model." in k]
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
                val = torch.nn.functional.cosine_similarity(t_a_flat.unsqueeze(0), t_b_flat.unsqueeze(0)).item()
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

    # Plot 1: Bar Chart
    fig_bar, ax = plt.subplots(figsize=(12, 6))

    colors = []
    for b in sorted_blocks:
        if "IN" in b:
            colors.append("#3498db")
        elif "MID" in b:
            colors.append("#f1c40f")
        elif "OUT" in b:
            colors.append("#e74c3c")
        elif "COND" in b:
            colors.append("#9b59b6")
        else:
            colors.append("#95a5a6")

    ax.bar(sorted_blocks, sorted_values, color=colors)
    ax.set_title(
        f"Model Difference Analysis ({metric})\n{model_a_path.split('/')[-1]} vs {model_b_path.split('/')[-1]}"
    )
    ax.set_ylabel(metric)
    ax.set_xlabel("UNet Blocks")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Plot 2: Heatmap (2D representation of data mapped into an arbitrary grid to show relative intensities)
    fig_heat, ax_heat = plt.subplots(figsize=(10, 8))

    # Create a 2D grid structure based on IN/MID/OUT categories and variations
    # Simple structuring for heatmap: rows = block families (IN, MID, OUT, COND), cols = layers indices
    heatmap_data = {}
    for b, val in zip(sorted_blocks, sorted_values):
        cat = "".join([c for c in b if not c.isdigit()])
        idx_str = "".join([c for c in b if c.isdigit()])
        idx = int(idx_str) if idx_str else 0

        if cat not in heatmap_data:
            heatmap_data[cat] = {}
        heatmap_data[cat][idx] = val

    all_indices = sorted(list(set(idx for cat_dict in heatmap_data.values() for idx in cat_dict.keys())))
    cats = sorted(list(heatmap_data.keys()))

    # Fill array
    heat_arr = np.zeros((len(cats), len(all_indices)))
    for i, c in enumerate(cats):
        for j, idx in enumerate(all_indices):
            heat_arr[i, j] = heatmap_data[c].get(idx, np.nan)  # nan where layer doesn't exist

    sns.heatmap(heat_arr, cmap="viridis", xticklabels=all_indices, yticklabels=cats, annot=True, fmt=".2f", ax=ax_heat)
    ax_heat.set_title(f"Block Differences Heatmap ({metric})")
    plt.tight_layout()

    # Plot 3: Radar Chart (Grouping into IN / MID / OUT / COND vectors)
    fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Aggregate values purely by IN, MID, OUT, COND categories for a neat radar
    radar_cats = ["IN", "MID", "OUT", "COND", "TIME", "OTHER"]
    radar_values = [0.0] * len(radar_cats)
    radar_counts = [0] * len(radar_cats)

    for b, val in zip(sorted_blocks, sorted_values):
        for i, c in enumerate(radar_cats):
            if c in b:
                radar_values[i] += val
                radar_counts[i] += 1
                break

    radar_means = [v / max(1, c) for v, c in zip(radar_values, radar_counts)]

    # Close the polygon by appending first value to the end
    angles = np.linspace(0, 2 * np.pi, len(radar_cats), endpoint=False).tolist()
    radar_means += radar_means[:1]
    angles += angles[:1]

    ax_radar.plot(angles, radar_means, color="#e74c3c", linewidth=2, linestyle="solid")
    ax_radar.fill(angles, radar_means, color="#e74c3c", alpha=0.4)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_cats)
    ax_radar.set_title(f"Structural Imbalance Radar ({metric})", y=1.1)

    return fig_bar, fig_heat, fig_radar
