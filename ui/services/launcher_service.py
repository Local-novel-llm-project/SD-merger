from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from module.logging_config import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_reflex_command(
    server_name: str = "0.0.0.0",
    server_port: int = 3000,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "reflex",
        "run",
        "--frontend-port",
        str(server_port),
        "--backend-port",
        str(server_port + 1),
    ]

    if server_name:
        command.extend(["--frontend-host", server_name, "--backend-host", server_name])

    return command


def launch_ui(
    server_name: str = "0.0.0.0",
    server_port: int = 3000,
    share: bool = False,
) -> None:
    if share:
        logger.warning("Reflex では Gradio share 相当の起動オプションは未対応です。share は無視します。")

    command = build_reflex_command(
        server_name=server_name,
        server_port=server_port,
    )
    logger.info("Starting Reflex UI: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
