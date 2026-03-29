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


def test_get_history_df_exposes_velocity_and_lrv(monkeypatch):
    monkeypatch.setattr(
        history_ui,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged.safetensors",
                "status": "Success",
                "config": {
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                            "left_right_velocity": 0.75,
                        }
                    ],
                },
            }
        ],
    )

    df = history_ui.get_history_df()

    assert df.iloc[0]["Velocity (Final)"] == 0.4
    assert df.iloc[0]["LRV (A/B)"] == 0.75


def test_get_history_df_resolves_auto_lrv_for_ab_entries(monkeypatch):
    monkeypatch.setattr(
        history_ui,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_ab.safetensors",
                "status": "Success",
                "config": {
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                        }
                    ],
                },
            }
        ],
    )

    df = history_ui.get_history_df()

    assert df.iloc[0]["Velocity (Final)"] == 0.4
    assert df.iloc[0]["LRV (A/B)"] == 0.4


def test_get_history_df_resolves_auto_lrv_for_target_entries(monkeypatch):
    monkeypatch.setattr(
        history_ui,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_target.safetensors",
                "status": "Success",
                "config": {
                    "target_model": "models/base.safetensors",
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                        }
                    ],
                },
            }
        ],
    )

    df = history_ui.get_history_df()

    assert df.iloc[0]["Velocity (Final)"] == 0.4
    assert df.iloc[0]["LRV (A/B)"] == 1.0


def test_get_history_df_supports_lr_alias(monkeypatch):
    monkeypatch.setattr(
        history_ui,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_alias.safetensors",
                "status": "Success",
                "config": {
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                            "lr": 0.9,
                        }
                    ],
                },
            }
        ],
    )

    df = history_ui.get_history_df()

    assert df.iloc[0]["LRV (A/B)"] == 0.9


def test_cleanup_download_file_removes_existing_file():
    runtime_dir = _make_runtime_dir("history_ui")
    recipe_path = runtime_dir / "recipe.yaml"
    recipe_path.write_text("models: []", encoding="utf-8")

    history_ui._cleanup_download_file(str(recipe_path))

    assert not recipe_path.exists()
