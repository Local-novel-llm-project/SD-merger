import glob
import importlib
import os
from typing import Any

def get_models_dir():
    """Returns the absolute path to the models directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "models")

def get_model_list():
    """Returns a list of relative model file paths found in the models directory."""
    models_dir = get_models_dir()
    if not os.path.exists(models_dir):
        return []

    model_files = []
    valid_extensions = (".safetensors", ".ckpt", ".pt", ".bin")

    for root, _, files in os.walk(models_dir, followlinks=True):
        for file in files:
            if file.endswith(valid_extensions):
                rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                model_files.append(rel_path)
                
    return sorted(model_files)

def get_model_path(model_name):
    """Converts a model name (relative path from dropdown) back to an absolute path."""
    if not model_name:
        return None
    return os.path.join(get_models_dir(), model_name)


def enqueue_merge_task(config: dict[str, Any], output_name: str, task_name: str) -> str:
    """Queues a merge-related task through the shared queue manager."""
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
    """Runs the merge pipeline directly from a config dictionary."""
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
