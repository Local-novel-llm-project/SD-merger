from pathlib import Path

import ui.components.history as history_ui


def _make_runtime_dir(name: str) -> Path:
    root = Path(__file__).parent / ".runtime"
    root.mkdir(exist_ok=True)
    path = root / name
    path.mkdir(exist_ok=True)
    return path


def test_get_history_df_handles_target_only_entries(monkeypatch):
    monkeypatch.setattr(
        history_ui,
        "load_history",
        lambda: [
            {
                "date": "2026-03-16 00:00:00",
                "output_name": "arthemy_tuned.safetensors",
                "status": "Success",
                "config": {
                    "target_model": "models/base.safetensors",
                    "models": [],
                },
            }
        ],
    )

    df = history_ui.get_history_df()

    assert len(df) == 1
    assert df.iloc[0]["Strategy"] == "target-only"
    assert df.iloc[0]["Model A"] == "models/base.safetensors"
    assert df.iloc[0]["Output Name"] == "arthemy_tuned.safetensors"


def test_cleanup_download_file_removes_existing_file():
    runtime_dir = _make_runtime_dir("history_ui")
    recipe_path = runtime_dir / "recipe.yaml"
    recipe_path.write_text("models: []", encoding="utf-8")

    history_ui._cleanup_download_file(str(recipe_path))

    assert not recipe_path.exists()
