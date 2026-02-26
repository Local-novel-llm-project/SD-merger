import os

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
