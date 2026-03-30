from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from module.generation import generate_image


@dataclass(frozen=True)
class ArtifactDTO:
    kind: str
    path: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRequest:
    model_path: str
    output_dir: str
    output_prefix: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.0
    sampler_name: str = "euler"
    scheduler: str = "normal"
    seed: int = -1
    artifact_kind: str = "image"
    artifact_label_prefix: str = "Generated"


@dataclass(frozen=True)
class GenerationResult:
    seed: int
    artifacts: tuple[ArtifactDTO, ...]
    preview: ArtifactDTO | None
    images: tuple[Image.Image, ...] = field(default_factory=tuple)


def resolve_generation_seed(seed: int) -> int:
    return seed if seed > 0 else random.randint(1, 1125899906842624)


def save_image_artifacts(
    images: Sequence[Image.Image],
    *,
    output_dir: str,
    output_prefix: str,
    artifact_kind: str = "image",
    artifact_label_prefix: str = "Generated",
    metadata_factory: Callable[[int], dict[str, Any]] | None = None,
) -> tuple[ArtifactDTO, ...]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    artifacts: list[ArtifactDTO] = []
    for index, image in enumerate(images):
        file_name = f"{output_prefix}_{timestamp}_{index}.png"
        artifact_path = os.path.join(output_dir, file_name)
        image.save(artifact_path)
        metadata = metadata_factory(index) if metadata_factory is not None else {}
        artifacts.append(
            ArtifactDTO(
                kind=artifact_kind,
                path=artifact_path,
                label=f"{artifact_label_prefix} {index + 1}",
                metadata=metadata,
            )
        )
    return tuple(artifacts)


def create_plot_artifact(
    image: Image.Image,
    *,
    output_dir: str,
    file_name: str,
    label: str,
    metadata: dict[str, Any] | None = None,
) -> ArtifactDTO:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    artifact_path = os.path.join(output_dir, file_name)
    image.save(artifact_path)
    return ArtifactDTO(
        kind="plot",
        path=artifact_path,
        label=label,
        metadata=metadata or {},
    )


def build_history_image_update(result: GenerationResult) -> dict[str, Any]:
    generated_paths = [artifact.path for artifact in result.artifacts if artifact.kind == "image"]
    if not generated_paths:
        return {}
    return {
        "preview_image": generated_paths[0],
        "generated_images": generated_paths,
    }


def generate_and_collect_artifacts(
    request: GenerationRequest,
    *,
    generate_fn: Callable[..., list[Image.Image] | None] = generate_image,
) -> GenerationResult:
    actual_seed = resolve_generation_seed(request.seed)
    generated = generate_fn(
        model_path=request.model_path,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg=request.cfg,
        sampler_name=request.sampler_name,
        scheduler=request.scheduler,
        seed=actual_seed,
    )

    images = tuple(generated or [])
    artifacts = save_image_artifacts(
        images,
        output_dir=request.output_dir,
        output_prefix=request.output_prefix,
        artifact_kind=request.artifact_kind,
        artifact_label_prefix=request.artifact_label_prefix,
        metadata_factory=lambda index: {
            "seed": actual_seed + index if request.seed > 0 else actual_seed,
            "prompt_index": index,
        },
    )
    preview = artifacts[0] if artifacts else None
    return GenerationResult(
        seed=actual_seed,
        artifacts=artifacts,
        preview=preview,
        images=images,
    )
