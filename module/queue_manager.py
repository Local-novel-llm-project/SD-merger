import os
import json
import time
import threading
import tempfile
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import deepcopy

from module.error_messages import build_user_error_message, build_user_error_summary


class QueueManager:
    """マージタスクのバックグラウンドキューを管理するシングルトンクラス"""

    _instance = None
    _lock = threading.Lock()

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
                        self.queue = data.get("queue", [])
                        self.is_paused = data.get("is_paused", False)

                        # 起動時に 'running' 状態のタスクがあれば 'error' に変更（クラッシュからの復帰）
                        for task in self.queue:
                            if task.get("status") == "running":
                                task["status"] = "error"
                                task["error"] = "Interrupted during previous run."

                except Exception as e:
                    logging.error(f"Failed to load queue.json: {e}")
                    self.queue = []
                    self.is_paused = False

    def save_queue(self):
        """キューの状態をディスクに永続化する"""
        with self._task_lock:
            try:
                data = {"queue": self.queue, "is_paused": self.is_paused}
                with open(self.queue_file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Failed to save queue.json: {e}")

    def add_task(self, config: Dict[str, Any], output_name: str, task_name: str = "Merge Task") -> str:
        """タスクをキューに追加する"""
        task_id = datetime.now().strftime("%Y%md%H%M%S") + f"_{len(self.queue)}"
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
        with self._task_lock:
            self.queue.append(task)
        self.save_queue()
        return task_id

    def remove_task(self, task_id: str) -> bool:
        """タスクをキューから削除する（実行中の場合は無視される）"""
        with self._task_lock:
            for i, task in enumerate(self.queue):
                if task["id"] == task_id:
                    if task["status"] == "running":
                        return False  # Cannot remove actively running task directly through this
                    self.queue.pop(i)
                    break
        self.save_queue()
        return True

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
        from module.history import save_history

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
            try:
                config = task_to_run["config"]

                # output_nameが含まれていなければ親からマージ
                if "output_name" not in config.get("models", [{}])[0] and "output_name" not in config:
                    if "models" in config and len(config["models"]) > 0:
                        config["models"][0]["output_name"] = task_to_run["output_name"]
                    else:
                        config["output_name"] = task_to_run["output_name"]

                if "poison_merge" in config:
                    from module.pipeline.poison import run_poison_merge

                    run_poison_merge(config, task_to_run["name"])
                else:
                    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
                        yaml.dump(config, f)
                        tmp_cfg = f.name

                    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "output"))

                    merger_main(tmp_cfg, out_dir)

                # history save if it's not a batch item saving logic from inside
                history_entry = {
                    "config": config,
                    "output_name": task_to_run["output_name"],
                    "status": "Success",
                }
                save_history(history_entry)
                success = True

            except Exception as e:
                error_msg = build_user_error_message(
                    e, action=f"{task_to_run['name']} の実行"
                )
                error_summary = build_user_error_summary(e)
                logging.exception("Task failed during queue execution.")
                # history save
                history_entry = {
                    "config": task_to_run["config"],
                    "output_name": task_to_run["output_name"],
                    "status": f"Failed: {error_summary}",
                }
                save_history(history_entry)

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
