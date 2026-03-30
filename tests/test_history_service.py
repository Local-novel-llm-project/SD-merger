import pytest

import module.services.history as history_service
import ui.services.history_service as ui_history_service


def test_build_history_rows_handles_target_only_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
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

    rows = history_service.build_history_rows()

    assert len(rows) == 1
    assert rows[0]["strategy"] == "target-only"
    assert rows[0]["model_a"] == "models/base.safetensors"
    assert rows[0]["output_name"] == "arthemy_tuned.safetensors"


def test_build_history_rows_exposes_velocity_and_lrv(monkeypatch):
    monkeypatch.setattr(
        history_service,
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

    rows = history_service.build_history_rows()

    assert rows[0]["velocity"] == "0.4"
    assert rows[0]["left_right_velocity"] == "0.75"


def test_build_history_rows_resolves_auto_lrv_for_ab_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
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

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "0.4"


def test_build_history_rows_resolves_auto_lrv_for_legacy_ab_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_legacy_ab.safetensors",
                "status": "Success",
                "config": {
                    "target_model": "models/a.safetensors",
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

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "0.4"


def test_build_history_rows_resolves_auto_lrv_for_target_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
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

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "1.0"


def test_build_history_rows_supports_lr_alias(monkeypatch):
    monkeypatch.setattr(
        history_service,
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

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "0.9"


def test_parse_history_yaml_text_validates_and_normalizes():
    config = history_service.parse_history_yaml_text(
        """
lazy_load: true
models:
  - left: models/a.safetensors
    right: models/b.safetensors
    strategy: mix
    velocity: 0.5
"""
    )

    assert config["lazy_load"] is True
    assert config["models"][0]["left"] == "models/a.safetensors"


def test_parse_history_yaml_text_rejects_invalid_yaml():
    with pytest.raises(ValueError, match="YAML parse error"):
        history_service.parse_history_yaml_text("models: [")


def test_load_yaml_from_path_reads_file(tmp_path):
    yaml_path = tmp_path / "recipe.yaml"
    yaml_path.write_text("models: []\n", encoding="utf-8")

    assert history_service.load_yaml_from_path(str(yaml_path)) == "models: []\n"


def test_export_yaml_to_path_writes_text(tmp_path):
    yaml_path = tmp_path / "nested" / "recipe.yaml"

    written_path = history_service.export_yaml_to_path("models: []\n", str(yaml_path))

    assert written_path == str(yaml_path)
    assert yaml_path.read_text(encoding="utf-8") == "models: []\n"


def test_build_yaml_download_payload_uses_output_name():
    data, filename = history_service.build_yaml_download_payload(
        "models: []\n",
        "merged.safetensors",
    )

    assert data == "models: []\n"
    assert filename == "merged.yaml"


def test_build_yaml_download_payload_sanitizes_filename():
    _, filename = history_service.build_yaml_download_payload(
        "models: []\n",
        "Upload: merged output.safetensors",
    )

    assert filename == "Upload_merged_output.yaml"


def test_queue_history_yaml_uses_resolved_output_name(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        history_service,
        "enqueue_merge_task",
        lambda config, output_name, task_name: calls.update(
            {
                "config": config,
                "output_name": output_name,
                "task_name": task_name,
            }
        )
        or "task-123",
    )

    task_id, output_name = history_service.queue_history_yaml(
        """
output_name: rerun_output.safetensors
models:
  - left: models/a.safetensors
    right: models/b.safetensors
    strategy: mix
    velocity: 0.5
"""
    )

    assert task_id == "task-123"
    assert output_name == "rerun_output.safetensors"
    assert calls["output_name"] == "rerun_output.safetensors"


def test_import_recipe_yaml_parses_mapping():
    recipe = history_service.import_recipe_yaml("models:\n  - left: a\n    right: b\n")

    assert recipe == {"models": [{"left": "a", "right": "b"}]}


def test_import_recipe_yaml_rejects_non_mapping():
    try:
        history_service.import_recipe_yaml("- just\n- a\n- list\n")
    except ValueError as exc:
        assert "mapping" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_ui_history_service_reexports_module_history_service():
    assert ui_history_service.build_history_rows is history_service.build_history_rows
    assert ui_history_service.queue_history_yaml is history_service.queue_history_yaml
    assert (
        ui_history_service.export_history_entry_to_path
        is history_service.export_history_entry_to_path
    )
