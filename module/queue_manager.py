import os
import json
import time
import threading
import tempfile
import yaml
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from copy import deepcopy

from module.error_messages import build_user_error_message, build_user_error_summary
from module.history import build_history_metadata, save_history, update_history_entry
from module.persistence import move_corrupt_file_aside, write_json_file_atomic


def _get_model_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = config.get("models")
    if not isinstance(models, list):
        return []
    return [model for model in models if isinstance(model, dict)]


def _resolve_task_output_name(config: Dict[str, Any], fallback_output_name: str) -> str:
    output_name = config.get("output_name")
    if output_name:
        return str(output_name)

    for model_config in _get_model_configs(config):
        model_output_name = model_config.get("output_name")
        if model_output_name:
            config["output_name"] = str(model_output_name)
            return config["output_name"]

    config["output_name"] = str(fallback_output_name)
    return config["output_name"]


def _create_task_id(queue_length: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{timestamp}_{queue_length}"


def _validate_loaded_task(task: Any, index: int) -> Dict[str, Any]:
    if not isinstance(task, dict):
        raise ValueError(f"queue[{index}] must be an object.")
    return task


def _extract_loaded_queue_state(data: Dict[str, Any]) -> tuple[List[Dict[str, Any]], bool]:
    queue_data = data.get("queue", [])
    if not isinstance(queue_data, list):
        raise ValueError("queue.json field 'queue' must be an array.")

    validated_queue = [
        _validate_loaded_task(task, index)
        for index, task in enumerate(queue_data)
    ]

    is_paused = data.get("is_paused", False)
    if not isinstance(is_paused, bool):
        raise ValueError("queue.json field 'is_paused' must be a boolean.")

    return validated_queue, is_paused


def _write_task_config_file(config: Dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        suffix=".yaml",
        encoding="utf-8",
    ) as config_file:
        yaml.safe_dump(config, config_file, allow_unicode=True, sort_keys=False)
        return config_file.name


def _remove_temp_file(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logging.warning(f"Failed to remove temporary config file '{path}': {exc}")


def _get_default_output_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "output")
    )


def _run_standard_merge_task(
    config: Dict[str, Any],
    merger_main: Callable[[str, str], Any],
    output_dir: str | None = None,
) -> None:
    temp_config_path = _write_task_config_file(config)
    try:
        merger_main(temp_config_path, output_dir or _get_default_output_dir())
    finally:
        _remove_temp_file(temp_config_path)


def _build_task_history_entry(
    config: Dict[str, Any],
    output_name: str,
    status: str,
) -> Dict[str, Any]:
    return {
        "config": deepcopy(config),
        "output_name": output_name,
        "status": status,
    }


def _record_task_history_start(config: Dict[str, Any], output_name: str) -> None:
    save_history(_build_task_history_entry(config, output_name, "Running"))


def _finalize_task_history(
    config: Dict[str, Any],
    output_name: str,
    *,
    success: bool,
    error_summary: str | None = None,
) -> None:
    final_status = "Success" if success else f"Failed: {error_summary or 'Unknown error'}"
    update_dict = {
        "config": deepcopy(config),
        "status": final_status,
        **build_history_metadata(),
    }
    if not update_history_entry(output_name, update_dict):
        save_history(_build_task_history_entry(config, output_name, final_status))


class QueueManager:
    """マージタスクのバックグラウンドキューを管理するシングルトンクラス"""

    _instance = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QueueManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, queue_file_path: Optional[str] = None):
        if self._initialized:
            return

        self.queue_file_path = queue_file_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "queue.json")
        )

        self.queue: List[Dict[str, Any]] = []
        self.is_paused = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._task_lock = threading.Lock()

        self.load_queue()
        self._initialized = True

    def load_queue(self):
        """ディスクからキューの状態を復元する"""
        with self._task_lock:
            if os.path.exists(self.queue_file_path):
                try:
                    with open(self.queue_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            raise ValueError("queue.json root must be an object.")
                        self.queue, self.is_paused = _extract_loaded_queue_state(data)

                        # 起動時に 'running' 状態のタスクがあれば 'error' に変更（クラッシュからの復帰）
                        for task in self.queue:
                            if task.get("status") == "running":
                                task["status"] = "error"
                                task["error"] = "Interrupted during previous run."

                except (json.JSONDecodeError, ValueError) as e:
                    moved_path = None
                    try:
                        moved_path = move_corrupt_file_aside(self.queue_file_path)
                    except OSError as move_exc:
                        logging.error(
                            "queue.json was corrupt but could not be moved aside: %s",
                            move_exc,
                        )
                    logging.error(
                        "Failed to load queue.json because it was corrupt. Moved to '%s': %s",
                        moved_path,
                        e,
                    )
                    self.queue = []
                    self.is_paused = False
                except OSError as e:
                    logging.error(f"Failed to load queue.json: {e}")
                    self.queue = []
                    self.is_paused = False

    def save_queue(self):
        """キューの状態をディスクに永続化する"""
        with self._task_lock:
            try:
                data = {"queue": self.queue, "is_paused": self.is_paused}
                write_json_file_atomic(
                    self.queue_file_path,
                    data,
                    indent=4,
                    ensure_ascii=False,
                )
            except Exception as e:
                logging.error(f"Failed to save queue.json: {e}")

    def add_task(self, config: Dict[str, Any], output_name: str, task_name: str = "Merge Task") -> str:
        """タスクをキューに追加する"""
        with self._task_lock:
            task_id = _create_task_id(len(self.queue))
            task = {
                "id": task_id,
                "name": task_name,
                "config": deepcopy(config),
                "output_name": output_name,
                "status": "pending",
                "progress": 0.0,
                "progress_desc": "Added to queue",
                "added_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None,
            }
            self.queue.append(task)
        self.save_queue()
        return task_id

    def remove_task(self, task_id: str) -> bool:
        """タスクをキューから削除する（実行中の場合は無視される）"""
        removed = False
        with self._task_lock:
            for i, task in enumerate(self.queue):
                if task["id"] == task_id:
                    if task["status"] == "running":
                        return False  # Cannot remove actively running task directly through this
                    self.queue.pop(i)
                    removed = True
                    break
        if removed:
            self.save_queue()
        return removed

    def clear_completed(self):
        """完了・エラーとなったタスクをキューから一括削除する"""
        with self._task_lock:
            self.queue = [t for t in self.queue if t["status"] in ("pending", "running")]
        self.save_queue()

    def get_queue(self) -> List[Dict[str, Any]]:
        """現在のキューのコピーを返す"""
        with self._task_lock:
            return deepcopy(self.queue)

    def pause(self):
        """キューの実行を一時停止（Running中のタスクは最後まで実行される）"""
        self.is_paused = True
        self.save_queue()

    def resume(self):
        """キューの実行を再開"""
        self.is_paused = False
        self.save_queue()

    def _worker_loop(self):
        """バックグラウンドで pending タスクを逐次実行するワーカー"""
        from main import main as merger_main

        logging.info("Queue worker thread started.")
        while not self._stop_event.is_set():
            if self.is_paused:
                time.sleep(2.0)
                continue

            # pendingタスクを探す
            task_to_run = None
            with self._task_lock:
                for task in self.queue:
                    if task["status"] == "pending":
                        task_to_run = task
                        task["status"] = "running"
                        task["progress"] = 0.0
                        task["progress_desc"] = "Starting..."
                        task["started_at"] = datetime.now().isoformat()
                        break

            if not task_to_run:
                time.sleep(2.0)
                continue

            self.save_queue()
            logging.info(f"Starting queued task: {task_to_run['name']} (ID: {task_to_run['id']})")

            success = False
            error_msg = None
            error_summary = None
            config = deepcopy(task_to_run["config"])
            resolved_output_name = task_to_run["output_name"]
            try:
                resolved_output_name = _resolve_task_output_name(
                    config, task_to_run["output_name"]
                )
                _record_task_history_start(config, resolved_output_name)

                if "poison_merge" in config:
                    from module.pipeline.poison import run_poison_merge

                    run_poison_merge(config, task_to_run["name"])
                else:
                    _run_standard_merge_task(config, merger_main)

                _finalize_task_history(
                    config,
                    resolved_output_name,
                    success=True,
                )
                success = True

            except Exception as e:
                error_msg = build_user_error_message(
                    e, action=f"{task_to_run['name']} の実行"
                )
                error_summary = build_user_error_summary(e)
                logging.exception("Task failed during queue execution.")
                _finalize_task_history(
                    config,
                    resolved_output_name,
                    success=False,
                    error_summary=error_summary,
                )

            finally:
                # 状態更新
                with self._task_lock:
                    for task in self.queue:
                        if task["id"] == task_to_run["id"]:
                            task["status"] = "completed" if success else "error"
                            task["progress"] = 1.0 if success else task.get("progress", 0.0)
                            task["progress_desc"] = (
                                "Completed"
                                if success
                                else error_summary or "Task failed."
                            )
                            task["error"] = error_msg
                            task["completed_at"] = datetime.now().isoformat()
                            break
                self.save_queue()

    def update_task_progress(self, task_id: str, progress: float, desc: str = ""):
        """指定したタスクの進捗と説明を更新する"""
        with self._task_lock:
            for task in self.queue:
                if task["id"] == task_id:
                    task["progress"] = max(0.0, min(1.0, float(progress)))
                    if desc:
                        task["progress_desc"] = desc
                    break
        # Option: periodically save or skip saving to reduce disk I/O during heavy processing
        # We save here to ensure UI can reflect the latest state across boundaries if needed
        self.save_queue()

    def start_worker(self):
        """ワーカーを起動する"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def stop_worker(self):
        """ワーカーを停止する"""
        if self._worker_thread is not None:
            self._stop_event.set()
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None


# グローバルインスタンス
queue_manager = QueueManager()
