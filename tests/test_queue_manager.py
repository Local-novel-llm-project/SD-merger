from pathlib import Path
import uuid

import pytest

from module import history
from module import queue_manager


def _make_runtime_dir(name: str) -> Path:
    root = Path(__file__).parent / ".runtime"
    root.mkdir(exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(exist_ok=True)
    return path


def test_resolve_task_output_name_promotes_nested_model_output_name():
    config = {
        "models": [
            {
                "left": "models/a.safetensors",
                "right": "models/b.safetensors",
                "output_name": "nested_output.safetensors",
            }
        ]
    }

    resolved = queue_manager._resolve_task_output_name(
        config,
        "fallback_output.safetensors",
    )

    assert resolved == "nested_output.safetensors"
    assert config["output_name"] == "nested_output.safetensors"


def test_resolve_task_output_name_supports_empty_models():
    config = {"target_model": "models/base.safetensors", "models": []}

    resolved = queue_manager._resolve_task_output_name(
        config,
        "target_only_output.safetensors",
    )

    assert resolved == "target_only_output.safetensors"
    assert config["output_name"] == "target_only_output.safetensors"


def test_extract_loaded_queue_state_accepts_valid_structure():
    queue, is_paused = queue_manager._extract_loaded_queue_state(
        {
            "queue": [{"id": "task-1", "status": "pending"}],
            "is_paused": False,
        }
    )

    assert isinstance(queue, list)
    assert queue[0]["id"] == "task-1"
    assert is_paused is False


def test_extract_loaded_queue_state_rejects_invalid_queue_shape():
    with pytest.raises(ValueError, match="queue"):
        queue_manager._extract_loaded_queue_state({"queue": {"id": "x"}})

    with pytest.raises(ValueError, match="queue\\[0\\]"):
        queue_manager._extract_loaded_queue_state({"queue": ["not-a-task"]})


def test_extract_loaded_queue_state_rejects_invalid_pause_flag():
    with pytest.raises(ValueError, match="is_paused"):
        queue_manager._extract_loaded_queue_state(
            {"queue": [], "is_paused": "false"}
        )


def test_run_standard_merge_task_cleans_temp_config():
    runtime_dir = _make_runtime_dir("queue_manager")
    observed = {}

    def fake_merger_main(config_path: str, output_dir: str) -> None:
        config_file = Path(config_path)
        observed["config_path"] = config_file
        observed["output_dir"] = output_dir
        assert config_file.exists()
        assert config_file.read_text(encoding="utf-8")

    queue_manager._run_standard_merge_task(
        {"target_model": "models/base.safetensors", "models": []},
        fake_merger_main,
        str(runtime_dir),
    )

    assert observed["output_dir"] == str(runtime_dir)
    assert not observed["config_path"].exists()


def test_remove_task_returns_false_for_missing_id(monkeypatch):
    manager = queue_manager.queue_manager
    original_queue = manager.queue
    manager.queue = []
    monkeypatch.setattr(manager, "save_queue", lambda: None)

    try:
        assert manager.remove_task("missing-id") is False
    finally:
        manager.queue = original_queue


def test_task_history_finalization_preserves_post_merge_updates(monkeypatch):
    runtime_dir = _make_runtime_dir("queue_history")
    history_file = runtime_dir / "merge_history.json"
    monkeypatch.setattr(history, "HISTORY_FILE", str(history_file))

    config = {"target_model": "models/base.safetensors", "models": []}
    output_name = "queued_output.safetensors"

    queue_manager._record_task_history_start(config, output_name)
    assert history.update_history_entry(
        output_name,
        {"preview_image": "preview.png"},
    )

    queue_manager._finalize_task_history(
        config,
        output_name,
        success=True,
    )

    entry = history.load_history()[0]
    assert entry["status"] == "Success"
    assert entry["preview_image"] == "preview.png"
