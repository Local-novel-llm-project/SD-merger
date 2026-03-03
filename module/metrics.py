import io
from typing import List, Dict
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

# CLIP Score calculation caching mechanism
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_clip_model():
    """Load CLIP model and processor lazily."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        logger.info("Loading CLIP model for metrics calculation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "openai/clip-vit-base-patch32"
        try:
            _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(device)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    return _CLIP_MODEL, _CLIP_PROCESSOR


def calculate_clip_score(images: List[Image.Image], prompt: str) -> List[float]:
    """
    Calculate CLIP score for a list of images against a prompt.

    Args:
        images: List of PIL.Image.Image to evaluate
        prompt: The text prompt to compare against

    Returns:
        List of float scores representing the similarity
    """
    if not images or not prompt:
        return []

    try:
        model, processor = get_clip_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Preprocess images and text
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # image_embeds and text_embeds are normalized by default
            # Logits per image is the dot product of normalized embeddings multiplied by a logit scale
            logits_per_image = outputs.logits_per_image  # this is exactly image_embeds @ text_embeds.t() * logit_scale
            # Alternatively, cosine similarity can be calculated manually or we can use the model logits as a scaled score.
            # Using raw cosine similarity for a straightforward 0-1 (or -1 to 1) score

            # Re-calculating raw cosine similarity for cleaner metrics
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # cosine similarity (N, 1) -> squeeze to (N,)
            cosine_sim = (image_embeds @ text_embeds.T).squeeze(-1)

        scores = cosine_sim.cpu().numpy().tolist()
        # Return as list, ensure it's a list even if there's only 1 image
        if not isinstance(scores, list):
            scores = [scores]
        return scores
    except Exception as e:
        logger.error(f"Error calculating CLIP score: {e}")
        return [0.0] * len(images)


def generate_radar_chart(metrics_data: Dict[str, Dict[str, float]], title: str = "A/B Test Metrics") -> Image.Image:
    """
    Generate a radar chart comparing multiple models on various metrics.

    Args:
        metrics_data: A dictionary where key is model/merge name, and value is a dict of metric_name -> score
        title: Title of the chart

    Returns:
        PIL.Image.Image containing the radar chart
    """
    if not metrics_data:
        # Return an empty image if no data
        return Image.new("RGB", (512, 512), color="white")

    # Extract labels (metric names)
    categories = []
    for model_stats in metrics_data.values():
        for metric in model_stats.keys():
            if metric not in categories:
                categories.append(metric)

    if not categories:
        return Image.new("RGB", (512, 512), color="white")

    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    # If there's only one or two categories, standard radar chart looks weird.
    # At least 3 points are needed for a polygon. We duplicate to form a triangle if N<3 (visual trick)
    # But pyplot handles it to some extent for N=1,2.

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color="grey", size=11)

    # Draw ylabels
    ax.set_rlabel_position(0)

    # Determine the min and max for the y-axis to make it consistent
    all_values = []
    for model_stats in metrics_data.values():
        all_values.extend(model_stats.values())

    max_val = max(all_values) if all_values else 1.0
    min_val = min(all_values) if all_values else 0.0

    # Pad limits a bit
    padding = (max_val - min_val) * 0.1
    if padding == 0:
        padding = 0.1

    plt.ylim(min_val - padding, max_val + padding)

    # Plot data
    for model_name, stats in metrics_data.items():
        values = [stats.get(cat, 0.0) for cat in categories]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=model_name)
        ax.fill(angles, values, alpha=0.1)

    # Add title and legend
    plt.title(title, size=14, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Save to a generic BytesIO
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf)
    return img
