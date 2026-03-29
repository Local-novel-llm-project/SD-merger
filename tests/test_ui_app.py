from pathlib import Path

from ui import app


def test_launch_ui_builds_reflex_command(monkeypatch):
    captured = {}

    def fake_run(command, check, cwd):
        captured["command"] = command
        captured["check"] = check
        captured["cwd"] = cwd

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    app.launch_ui(server_name="127.0.0.1", server_port=3100, share=False)

    assert captured["check"] is True
    assert captured["cwd"] == app.PROJECT_ROOT
    assert captured["command"][:4] == [app.sys.executable, "-m", "reflex", "run"]
    assert "--frontend-port" in captured["command"]
    assert "3100" in captured["command"]
    assert "--backend-port" in captured["command"]
    assert "3101" in captured["command"]


def test_launch_ui_logs_warning_when_share_requested(monkeypatch):
    warnings = []

    monkeypatch.setattr(app.subprocess, "run", lambda command, check, cwd: None)
    monkeypatch.setattr(app.logger, "warning", lambda message: warnings.append(message))

    app.launch_ui(share=True)

    assert warnings
    assert "share" in warnings[0]


def test_project_root_points_to_repository_root():
    assert Path(app.PROJECT_ROOT, "main.py").exists()
    assert Path(app.PROJECT_ROOT, "rxconfig.py").exists()
