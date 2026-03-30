from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from main import run_merge_pipeline
from module.metrics import calculate_clip_score, generate_radar_chart
from module.services.generation import (
    ArtifactDTO,
    GenerationRequest,
    GenerationResult,
    create_plot_artifact,
    generate_and_collect_artifacts,
)
from module.services.merge import build_basic_merge_config


@dataclass(frozen=True)
class AbTestRunResult:
    model_order: str
    output_model_path: str
    merge_config: dict[str, Any]
    generation: GenerationResult
    metrics: dict[str, float]


@dataclass(frozen=True)
class AbTestResult:
    runs: tuple[AbTestRunResult, ...]
    metrics: dict[str, dict[str, float]]
    artifacts: tuple[ArtifactDTO, ...]


def build_ab_test_configs(
    *,
    model_a: str,
    model_b: str,
    model_c: str | None,
    strategy: str,
    target_strategy: str,
    velocity: float,
    left_right_velocity: str | None,
    use_advanced_options: bool,
    mbw: str | None,
    bake_in_vae: str | None,
    lazy_load: bool,
    output_name_prefix: str = "ab_test",
) -> dict[str, tuple[dict[str, Any], str]]:
    order_pairs = {
        "A->B": (model_a, model_b),
        "B->A": (model_b, model_a),
    }
    configs: dict[str, tuple[dict[str, Any], str]] = {}
    for model_order, (left_model, right_model) in order_pairs.items():
        config, resolved_name = build_basic_merge_config(
            left_model,
            right_model,
            model_c,
            strategy,
            target_strategy,
            velocity,
            left_right_velocity,
            use_advanced_options,
            mbw,
            bake_in_vae,
            f"{output_name_prefix}_{model_order.replace('->', '_')}.safetensors",
            lazy_load,
        )
        configs[model_order] = (config, resolved_name)
    return configs


def run_ab_test(
    *,
    model_a: str,
    model_b: str,
    prompt: str,
    output_dir: str,
    model_c: str | None = None,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    seed: int = -1,
    strategy: str = "mix",
    target_strategy: str = "mix",
    velocity: float = 0.5,
    left_right_velocity: str | None = None,
    use_advanced_options: bool = False,
    mbw: str | None = None,
    bake_in_vae: str | None = None,
    lazy_load: bool = True,
    merge_runner: Callable[[dict[str, Any], str], str | None] = run_merge_pipeline,
    generation_runner: Callable[[GenerationRequest], GenerationResult] = generate_and_collect_artifacts,
    clip_score_fn: Callable[[list[Any], str], list[float]] = calculate_clip_score,
    radar_chart_fn: Callable[[dict[str, dict[str, float]], str], Any] = generate_radar_chart,
) -> AbTestResult:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    configs = build_ab_test_configs(
        model_a=model_a,
        model_b=model_b,
        model_c=model_c,
        strategy=strategy,
        target_strategy=target_strategy,
        velocity=velocity,
        left_right_velocity=left_right_velocity,
        use_advanced_options=use_advanced_options,
        mbw=mbw,
        bake_in_vae=bake_in_vae,
        lazy_load=lazy_load,
    )

    runs: list[AbTestRunResult] = []
    metrics: dict[str, dict[str, float]] = {}

    for model_order, (config, resolved_name) in configs.items():
        merged_model_path = merge_runner(config, output_dir)
        if not merged_model_path:
            raise ValueError(f"Merge runner returned no output for {model_order}.")

        generation = generation_runner(
            GenerationRequest(
                model_path=merged_model_path,
                output_dir=output_dir,
                output_prefix=os.path.splitext(resolved_name)[0] + "_sample",
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                seed=seed,
                artifact_label_prefix=model_order,
            )
        )
        clip_scores = clip_score_fn(list(generation.images), prompt)
        run_metrics = {
            "clip_score_mean": sum(clip_scores) / len(clip_scores) if clip_scores else 0.0,
            "image_count": float(len(generation.artifacts)),
        }
        metrics[model_order] = run_metrics
        runs.append(
            AbTestRunResult(
                model_order=model_order,
                output_model_path=merged_model_path,
                merge_config=config,
                generation=generation,
                metrics=run_metrics,
            )
        )

    artifacts: list[ArtifactDTO] = [
        artifact
        for run in runs
        for artifact in run.generation.artifacts
    ]
    if metrics:
        radar_image = radar_chart_fn(metrics, "A/B Test Metrics")
        artifacts.append(
            create_plot_artifact(
                radar_image,
                output_dir=output_dir,
                file_name="ab_test_metrics.png",
                label="A/B Test Metrics",
                metadata={"metric_names": sorted(next(iter(metrics.values())).keys())},
            )
        )

    return AbTestResult(
        runs=tuple(runs),
        metrics=metrics,
        artifacts=tuple(artifacts),
    )
