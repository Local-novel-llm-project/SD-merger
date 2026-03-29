from __future__ import annotations

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


def is_paused() -> bool:
    return bool(queue_manager.is_paused)


def pause_queue() -> None:
    queue_manager.pause()


def resume_queue() -> None:
    queue_manager.resume()


def clear_completed() -> None:
    queue_manager.clear_completed()


def remove_task(task_id: str) -> bool:
    return queue_manager.remove_task(task_id)
