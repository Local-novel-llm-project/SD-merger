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


def test_update_history_entry_matches_basename_and_stem(monkeypatch):
    runtime_dir = _make_runtime_dir("history_normalized")
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

        updated = history.update_history_entry(
            str(runtime_dir / "merged_a.safetensors"),
            {"preview_image": "preview.png"},
        )
        assert updated is True

        updated = history.update_history_entry(
            "merged_a",
            {"generated_images": ["sample.png"]},
        )
        assert updated is True
        entry = history.load_history()[0]
        assert entry["preview_image"] == "preview.png"
        assert entry["generated_images"] == ["sample.png"]
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_update_history_entry_prefers_exact_name_over_same_stem(monkeypatch):
    runtime_dir = _make_runtime_dir("history_exact_match")
    history_file = runtime_dir / "merge_history.json"
    monkeypatch.setattr(history, "HISTORY_FILE", str(history_file))

    try:
        history.save_history(
            {
                "config": {"models": [{"left": "a", "right": "b"}]},
                "output_name": "foo.safetensors",
                "status": "Success",
            }
        )
        history.save_history(
            {
                "config": {"models": [{"left": "a", "right": "b"}]},
                "output_name": "foo.ckpt",
                "status": "Success",
            }
        )

        updated = history.update_history_entry(
            "foo.safetensors",
            {"preview_image": "preview.png"},
        )

        assert updated is True
        entries = history.load_history()
        assert entries[0].get("preview_image") is None
        assert entries[1]["preview_image"] == "preview.png"
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_update_history_entry_updates_most_recent_duplicate_exact_path(monkeypatch):
    runtime_dir = _make_runtime_dir("history_exact_duplicate")
    history_file = runtime_dir / "merge_history.json"
    monkeypatch.setattr(history, "HISTORY_FILE", str(history_file))

    try:
        for index in range(2):
            history.save_history(
                {
                    "config": {"models": [{"left": "a", "right": "b"}]},
                    "output_name": "same_output.safetensors",
                    "status": f"Success-{index}",
                }
            )

        updated = history.update_history_entry(
            "same_output.safetensors",
            {"preview_image": "latest.png"},
        )

        assert updated is True
        entries = history.load_history()
        assert entries[0]["preview_image"] == "latest.png"
        assert entries[1].get("preview_image") is None
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_load_history_moves_corrupt_file_aside(monkeypatch):
    runtime_dir = _make_runtime_dir("history_corrupt")
    history_file = runtime_dir / "merge_history.json"
    history_file.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(history, "HISTORY_FILE", str(history_file))

    try:
        assert history.load_history() == []
        assert not history_file.exists()
        moved_files = list(runtime_dir.glob("merge_history.corrupt.*.json"))
        assert len(moved_files) == 1
        assert moved_files[0].read_text(encoding="utf-8") == "{invalid"
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_update_history_entry_rejects_ambiguous_basename_matches(monkeypatch):
    runtime_dir = _make_runtime_dir("history_basename_collision")
    history_file = runtime_dir / "merge_history.json"
    monkeypatch.setattr(history, "HISTORY_FILE", str(history_file))

    first_output = runtime_dir / "dir1" / "foo.safetensors"
    second_output = runtime_dir / "dir2" / "foo.safetensors"

    try:
        history.save_history(
            {
                "config": {"models": [{"left": "a", "right": "b"}]},
                "output_name": str(first_output),
                "status": "Success",
            }
        )
        history.save_history(
            {
                "config": {"models": [{"left": "a", "right": "b"}]},
                "output_name": str(second_output),
                "status": "Success",
            }
        )

        updated = history.update_history_entry(
            "foo.safetensors",
            {"preview_image": "preview.png"},
        )

        assert updated is False
        assert all(
            entry.get("preview_image") is None for entry in history.load_history()
        )
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
