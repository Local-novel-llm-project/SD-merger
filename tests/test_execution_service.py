import os
import sys
import types

import module.services.merge as merge_services
import ui.services.execution_service as execution_service
import ui.utils as utils


def test_enqueue_merge_task_delegates_to_queue_manager(monkeypatch):
    calls = {}

    class FakeQueueManager:
        def add_task(self, config, output_name, task_name=None):
            calls["config"] = config
            calls["output_name"] = output_name
            calls["task_name"] = task_name
            return "task-123"

    fake_queue_module = types.ModuleType("module.queue_manager")
    fake_queue_module.queue_manager = FakeQueueManager()
    monkeypatch.setitem(sys.modules, "module.queue_manager", fake_queue_module)

    task_id = merge_services.enqueue_merge_task(
        {"models": [{"left": "a", "right": "b"}]},
        "merged.safetensors",
        "Smoke Test",
    )

    assert task_id == "task-123"
    assert calls == {
        "config": {"models": [{"left": "a", "right": "b"}]},
        "output_name": "merged.safetensors",
        "task_name": "Smoke Test",
    }


def test_run_merge_from_config_loads_extensions_and_cleans_stale_outputs(monkeypatch):
    calls = {"load_extensions": 0}
    removed_files = []
    created_dirs = []
    requested_output_dir = os.path.abspath("merged/test_execution_service")
    stale_output = os.path.join(requested_output_dir, "stale.safetensors")
    expected_output = os.path.join(requested_output_dir, "fresh.safetensors")

    fake_extension_module = types.ModuleType("module.extension_manager")

    def fake_load_extensions():
        calls["load_extensions"] += 1

    fake_extension_module.load_extensions = fake_load_extensions

    fake_main_module = types.ModuleType("main")

    def fake_run_merge_pipeline(config, default_output_dir):
        calls["config"] = config
        calls["default_output_dir"] = default_output_dir
        return expected_output

    fake_main_module.run_merge_pipeline = fake_run_merge_pipeline

    monkeypatch.setitem(
        sys.modules,
        "module.extension_manager",
        fake_extension_module,
    )
    monkeypatch.setitem(sys.modules, "main", fake_main_module)
    monkeypatch.setattr(
        merge_services.queue_tasks.os,
        "makedirs",
        lambda path, exist_ok=True: created_dirs.append((path, exist_ok)),
    )
    monkeypatch.setattr(
        merge_services.queue_tasks.glob,
        "glob",
        lambda pattern: [stale_output],
    )
    monkeypatch.setattr(
        merge_services.queue_tasks.os.path,
        "isfile",
        lambda path: path == stale_output,
    )
    monkeypatch.setattr(
        merge_services.queue_tasks.os,
        "remove",
        lambda path: removed_files.append(path),
    )

    result = merge_services.run_merge_from_config(
        {"target_model": "model_a.safetensors", "models": []},
        "merged/test_execution_service",
    )

    assert result == expected_output
    assert calls["load_extensions"] == 1
    assert calls["default_output_dir"] == requested_output_dir
    assert created_dirs == [(requested_output_dir, True)]
    assert removed_files == [stale_output]


def test_ui_utils_reexports_execution_helpers():
    assert execution_service.enqueue_merge_task is merge_services.enqueue_merge_task
    assert execution_service.run_merge_from_config is merge_services.run_merge_from_config
    assert utils.enqueue_merge_task is merge_services.enqueue_merge_task
    assert utils.run_merge_from_config is merge_services.run_merge_from_config
