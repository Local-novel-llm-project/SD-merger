import shutil
import uuid
from pathlib import Path

import yaml

from module import history


def _make_runtime_dir(name: str) -> Path:
    root = Path(__file__).parent / ".runtime"
    root.mkdir(exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_save_history_and_update_entry(monkeypatch):
    runtime_dir = _make_runtime_dir("history")
    history_file = runtime_dir / "merge_history.json"
    monkeypatch.setattr(history, "HISTORY_FILE", str(history_file))

    try:
        history.save_history(
            {
                "config": {"models": [{"left": "a", "right": "b"}]},
                "output_name": "merged_a.safetensors",
                "status": "Success",
            }
        )

        entries = history.load_history()
        assert len(entries) == 1
        assert entries[0]["output_name"] == "merged_a.safetensors"
        assert "timestamp" in entries[0]
        assert "date" in entries[0]

        updated = history.update_history_entry(
            "merged_a.safetensors",
            {"preview_image": "preview.png"},
        )
        assert updated is True
        assert history.load_history()[0]["preview_image"] == "preview.png"
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_export_recipe_writes_yaml_with_metadata(monkeypatch):
    runtime_dir = _make_runtime_dir("recipe")
    history_file = runtime_dir / "merge_history.json"
    output_file = runtime_dir / "recipe.yaml"
    monkeypatch.setattr(history, "HISTORY_FILE", str(history_file))

    entry = {
        "date": "2026-03-11 12:00:00",
        "output_name": "merged_model.safetensors",
        "status": "Success",
        "config": {
            "models": [
                {
                    "left": "a.safetensors",
                    "right": "b.safetensors",
                    "strategy": "mix",
                }
            ]
        },
    }

    try:
        history.export_recipe(entry, str(output_file))
        content = output_file.read_text(encoding="utf-8")

        assert "# Auto-generated merge recipe from history" in content
        assert "# Original Output Name: merged_model.safetensors" in content

        yaml_body = "\n".join(
            line for line in content.splitlines() if not line.startswith("#")
        )
        parsed = yaml.safe_load(yaml_body)
        assert parsed["models"][0]["strategy"] == "mix"
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
