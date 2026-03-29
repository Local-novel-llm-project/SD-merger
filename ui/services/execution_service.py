from __future__ import annotations

import glob
import importlib
import os
from typing import Any


def enqueue_merge_task(config: dict[str, Any], output_name: str, task_name: str) -> str:
    """Queue a merge-related task through the shared queue manager."""
    queue_module = importlib.import_module("module.queue_manager")
    return queue_module.queue_manager.add_task(
        config,
        output_name,
        task_name=task_name,
    )


def run_merge_from_config(
    config: dict[str, Any],
    output_dir: str,
    cleanup_pattern: str = "*.safetensors",
) -> str | None:
    """Run the merge pipeline directly from a config dictionary."""
    extension_module = importlib.import_module("module.extension_manager")
    main_module = importlib.import_module("main")

    resolved_output_dir = os.path.abspath(output_dir)
    os.makedirs(resolved_output_dir, exist_ok=True)

    if cleanup_pattern:
        for stale_file in glob.glob(os.path.join(resolved_output_dir, cleanup_pattern)):
            if os.path.isfile(stale_file):
                os.remove(stale_file)

    extension_module.load_extensions()
    return main_module.run_merge_pipeline(
        config,
        default_output_dir=resolved_output_dir,
    )
