import shutil
import uuid
from pathlib import Path

from module import presets


def _make_runtime_dir(name: str) -> Path:
    root = Path(__file__).parent / ".runtime"
    root.mkdir(exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_save_and_list_and_load_preset(monkeypatch):
    runtime_dir = _make_runtime_dir("presets")
    monkeypatch.setattr(presets, "PRESET_DIR", str(runtime_dir))

    try:
        save_result = presets.save_preset(
            "portrait",
            {"prompt": "portrait", "steps": 30},
            category="generation",
        )

        assert "portrait.json" in save_result
        assert presets.list_presets("generation") == ["portrait.json"]
        assert presets.load_preset("portrait.json", "generation") == {
            "prompt": "portrait",
            "steps": 30,
        }
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_load_preset_returns_empty_dict_for_invalid_json(monkeypatch):
    runtime_dir = _make_runtime_dir("broken_presets")
    category_dir = runtime_dir / "default"
    category_dir.mkdir()
    (category_dir / "broken.json").write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(presets, "PRESET_DIR", str(runtime_dir))

    try:
        assert presets.load_preset("broken.json") == {}
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
