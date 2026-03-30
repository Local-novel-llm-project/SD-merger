import module.services.queue as queue_service
import ui.services.queue_service as ui_queue_service


def test_build_queue_rows_formats_progress_and_description(monkeypatch):
    monkeypatch.setattr(
        queue_service.queue_manager,
        "get_queue",
        lambda: [
            {
                "id": "task-1",
                "name": "Merge Task",
                "status": "pending",
                "output_name": "merged.safetensors",
                "progress": 0.25,
                "progress_desc": "Queued",
            }
        ],
    )

    rows = queue_service.build_queue_rows()

    assert rows == [
        {
            "id": "task-1",
            "name": "Merge Task",
            "status": "pending",
            "output_name": "merged.safetensors",
            "progress": "25%",
            "desc": "Queued",
        }
    ]


def test_build_queue_snapshot_exposes_pause_state(monkeypatch):
    monkeypatch.setattr(
        queue_service.queue_manager,
        "get_queue",
        lambda: [{"id": "task-1", "status": "pending"}],
    )
    monkeypatch.setattr(queue_service.queue_manager, "is_paused", True)

    snapshot = queue_service.build_queue_snapshot()

    assert snapshot["paused"] is True
    assert snapshot["task_count"] == 1
    assert len(snapshot["rows"]) == 1


def test_remove_task_rejects_running_task(monkeypatch):
    monkeypatch.setattr(
        queue_service.queue_manager,
        "get_queue",
        lambda: [{"id": "task-1", "status": "running"}],
    )

    ok, message = queue_service.remove_task("task-1")

    assert ok is False
    assert "running" in message


def test_remove_task_returns_success_when_manager_removes(monkeypatch):
    monkeypatch.setattr(
        queue_service.queue_manager,
        "get_queue",
        lambda: [{"id": "task-1", "status": "pending"}],
    )
    monkeypatch.setattr(queue_service.queue_manager, "remove_task", lambda task_id: task_id == "task-1")

    ok, message = queue_service.remove_task("task-1")

    assert ok is True
    assert message == "Removed task: task-1"


def test_ui_queue_service_reexports_module_queue_service():
    assert ui_queue_service.build_queue_rows is queue_service.build_queue_rows
    assert ui_queue_service.build_queue_snapshot is queue_service.build_queue_snapshot
    assert ui_queue_service.remove_task is queue_service.remove_task
