from pathlib import Path

from ui import app
from ui.services import launcher_service


def test_build_reflex_command_includes_ports_and_host():
    command = launcher_service.build_reflex_command(
        server_name="127.0.0.1",
        server_port=3100,
    )

    assert command[:4] == [launcher_service.sys.executable, "-m", "reflex", "run"]
    assert "--frontend-port" in command
    assert "3100" in command
    assert "--backend-port" in command
    assert "3101" in command
    assert "--frontend-host" in command
    assert "--backend-host" in command


def test_launch_ui_builds_reflex_command(monkeypatch):
    captured = {}

    def fake_run(command, check, cwd):
        captured["command"] = command
        captured["check"] = check
        captured["cwd"] = cwd

    monkeypatch.setattr(launcher_service.subprocess, "run", fake_run)

    launcher_service.launch_ui(server_name="127.0.0.1", server_port=3100, share=False)

    assert captured["check"] is True
    assert captured["cwd"] == launcher_service.PROJECT_ROOT
    assert captured["command"][:4] == [
        launcher_service.sys.executable,
        "-m",
        "reflex",
        "run",
    ]


def test_launch_ui_logs_warning_when_share_requested(monkeypatch):
    warnings = []

    monkeypatch.setattr(
        launcher_service.subprocess,
        "run",
        lambda command, check, cwd: None,
    )
    monkeypatch.setattr(
        launcher_service.logger,
        "warning",
        lambda message: warnings.append(message),
    )

    launcher_service.launch_ui(share=True)

    assert warnings
    assert "share" in warnings[0]


def test_app_reexports_launcher_service_symbols():
    assert app.launch_ui is launcher_service.launch_ui
    assert app.PROJECT_ROOT == launcher_service.PROJECT_ROOT


def test_project_root_points_to_repository_root():
    assert Path(app.PROJECT_ROOT, "main.py").exists()
    assert Path(app.PROJECT_ROOT, "rxconfig.py").exists()
