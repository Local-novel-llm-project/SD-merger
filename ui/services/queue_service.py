from __future__ import annotations

from typing import Any

from module.queue_manager import queue_manager


def build_queue_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for task in queue_manager.get_queue():
        rows.append(
            {
                "id": str(task.get("id") or ""),
                "name": str(task.get("name") or ""),
                "status": str(task.get("status") or ""),
                "output_name": str(task.get("output_name") or ""),
                "progress": f"{float(task.get('progress', 0.0)) * 100:.0f}%",
                "desc": str(task.get("progress_desc") or ""),
            }
        )

    return rows


def get_queue_task(task_id: str) -> dict[str, Any] | None:
    for task in queue_manager.get_queue():
        if str(task.get("id") or "") == task_id:
            return task
    return None


def build_queue_snapshot() -> dict[str, Any]:
    rows = build_queue_rows()
    return {
        "rows": rows,
        "paused": bool(queue_manager.is_paused),
        "task_count": len(rows),
    }


def is_paused() -> bool:
    return bool(queue_manager.is_paused)


def pause_queue() -> tuple[bool, str]:
    queue_manager.pause()
    return True, "Queue paused."


def resume_queue() -> tuple[bool, str]:
    queue_manager.resume()
    return True, "Queue resumed."


def clear_completed() -> tuple[bool, str]:
    queue_manager.clear_completed()
    return True, "Completed and errored tasks were cleared."


def remove_task(task_id: str) -> tuple[bool, str]:
    task = get_queue_task(task_id)
    if task is None:
        return False, f"Task was not found: {task_id}"
    if str(task.get("status") or "") == "running":
        return False, f"Task is running and cannot be removed: {task_id}"
    if queue_manager.remove_task(task_id):
        return True, f"Removed task: {task_id}"
    return False, f"Task could not be removed: {task_id}"
