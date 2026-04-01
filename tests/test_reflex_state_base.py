from __future__ import annotations

import importlib
import sys
import types


def _install_fake_reflex(monkeypatch):
    fake_reflex = types.ModuleType("reflex")

    class FakeState:
        def __init__(self):
            self.router = types.SimpleNamespace(page=types.SimpleNamespace(path=""))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def event(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    fake_reflex.State = FakeState
    fake_reflex.Component = object
    fake_reflex.Var = str
    fake_reflex.UploadFile = object
    fake_reflex.event = event
    fake_reflex.var = lambda func: property(func)
    fake_reflex.download = lambda **kwargs: {"type": "download", **kwargs}
    fake_reflex.clear_selected_files = (
        lambda upload_id: {"type": "clear_selected_files", "upload_id": upload_id}
    )

    monkeypatch.setitem(sys.modules, "reflex", fake_reflex)
    return fake_reflex


def _reload_state_modules():
    for module_name in [
        "ui.state.base",
        "ui.state.merge_state",
        "ui.state.tune_state",
        "ui.state.queue_state",
        "ui.state.history_state",
    ]:
        sys.modules.pop(module_name, None)

    return {
        "base": importlib.import_module("ui.state.base"),
        "merge": importlib.import_module("ui.state.merge_state"),
        "tune": importlib.import_module("ui.state.tune_state"),
        "queue": importlib.import_module("ui.state.queue_state"),
        "history": importlib.import_module("ui.state.history_state"),
    }


def test_base_page_state_formats_user_friendly_errors(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    state = modules["base"].BasePageState()

    state.set_error(FileNotFoundError("missing.safetensors"), action="モデル一覧更新")

    assert state.status_variant == "error"
    assert "モデル一覧更新に失敗しました。" in state.status_message
    assert "必要なモデルまたはファイルが見つかりません。" in state.status_message


def test_refresh_models_updates_models_and_reconciles_selection(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    merge_module = modules["merge"]
    base_module = modules["base"]

    monkeypatch.setattr(base_module, "ensure_app_ready", lambda: None)
    monkeypatch.setattr(base_module, "list_models", lambda: ["b.safetensors"])

    state = merge_module.MergeState()
    state.model_a = "a.safetensors"
    state.model_b = "b.safetensors"
    state.model_c = "c.safetensors"
    state.bake_in_vae = "vae.safetensors"

    state.refresh_models()

    assert state.available_models == ["b.safetensors"]
    assert state.model_a == ""
    assert state.model_b == "b.safetensors"
    assert state.model_c == "選択しない"
    assert state.bake_in_vae == ""
    assert state.status_variant == "info"
    assert state.busy is False


def test_refresh_models_reports_user_friendly_error(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    base_module = modules["base"]
    monkeypatch.setattr(base_module, "list_models", lambda: (_ for _ in ()).throw(FileNotFoundError("models")))

    state = modules["tune"].TuneState()
    state.refresh_models()

    assert state.available_models == []
    assert state.status_variant == "error"
    assert "必要なモデルまたはファイルが見つかりません。" in state.status_message


def test_queue_state_load_page_refreshes_and_starts_polling(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    queue_module = modules["queue"]
    base_module = modules["base"]

    calls = {"ready": 0, "refresh": 0, "poll": 0}
    monkeypatch.setattr(base_module, "ensure_app_ready", lambda: calls.__setitem__("ready", calls["ready"] + 1))
    monkeypatch.setattr(
        queue_module.QueueState,
        "refresh",
        lambda self: calls.__setitem__("refresh", calls["refresh"] + 1),
    )
    monkeypatch.setattr(
        queue_module.QueueState,
        "start_polling",
        lambda self: calls.__setitem__("poll", calls["poll"] + 1) or "poll-event",
    )

    state = queue_module.QueueState()
    result = state.load_page()

    assert result == "poll-event"
    assert calls == {"ready": 1, "refresh": 1, "poll": 1}


def test_history_state_upload_uses_shared_helper(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    history_module = modules["history"]
    base_module = modules["base"]
    monkeypatch.setattr(base_module, "ensure_app_ready", lambda: None)

    class FakeUpload:
        filename = "recipe.yaml"

        async def read(self):
            return b"models: []\n"

    state = history_module.HistoryState()
    result = __import__("asyncio").run(state.handle_yaml_upload([FakeUpload()]))

    assert state.yaml_editor_text == "models: []\n"
    assert state.yaml_preview == "models: []\n"
    assert state.yaml_source_label == "Upload: recipe.yaml"
    assert state.status_variant == "success"
    assert result == {
        "type": "clear_selected_files",
        "upload_id": history_module.HISTORY_YAML_UPLOAD_ID,
    }


def test_history_state_download_uses_shared_download_helper(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    history_module = modules["history"]

    state = history_module.HistoryState()
    state.yaml_editor_text = "models: []\n"
    state.selected_output_name = "merged.safetensors"

    result = state.download_yaml()

    assert result == {"type": "download", "data": "models: []\n", "filename": "merged.yaml"}
    assert state.status_variant == "info"
    assert state.status_message == "Downloading YAML: merged.yaml"


def test_merge_state_updates_output_name_until_user_overrides(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    merge_module = modules["merge"]

    monkeypatch.setattr(merge_module, "create_default_output_name", lambda a, b: f"{a}+{b}")
    monkeypatch.setattr(merge_module, "build_merge_preview", lambda *args: "{\"preview\": true}")

    state = merge_module.MergeState()

    state.set_model_a_value("ModelA")
    state.set_model_b_value("ModelB")
    assert state.output_name == "ModelA+ModelB"
    assert state.preview_json == "{\"preview\": true}"

    state.set_output_name_value("custom-name")
    state.set_model_a_value("ModelA2")
    assert state.output_name == "custom-name"


def test_merge_state_queue_current_merge_uses_strategy_task_name(monkeypatch):
    _install_fake_reflex(monkeypatch)
    modules = _reload_state_modules()
    merge_module = modules["merge"]

    captured = {}

    monkeypatch.setattr(
        merge_module,
        "build_basic_merge_config",
        lambda *args: ({"models": [{"strategy": "mix"}]}, "merged.safetensors"),
    )
    monkeypatch.setattr(
        merge_module,
        "queue_merge",
        lambda config, output_name, task_name=None: captured.update(
            {
                "config": config,
                "output_name": output_name,
                "task_name": task_name,
            }
        )
        or "task-123",
    )
    monkeypatch.setattr(merge_module, "build_preview_json", lambda config: "{\"queued\": true}")

    state = merge_module.MergeState()
    state.strategy = "tensor"
    state.velocity = "0.5"

    state.queue_current_merge()

    assert captured["config"] == {"models": [{"strategy": "mix"}]}
    assert captured["output_name"] == "merged.safetensors"
    assert captured["task_name"] == "Merge: tensor"
    assert state.last_task_id == "task-123"
    assert state.preview_json == "{\"queued\": true}"
    assert state.status_variant == "success"
