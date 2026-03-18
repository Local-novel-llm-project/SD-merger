import json
import os
import tempfile
from datetime import datetime
from typing import Any


def write_json_file_atomic(
    path: str,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".tmp_",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=indent, ensure_ascii=ensure_ascii)
            file.flush()
            if hasattr(os, "fsync"):
                os.fsync(file.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def move_corrupt_file_aside(path: str, *, label: str = "corrupt") -> str | None:
    if not os.path.exists(path):
        return None

    base, ext = os.path.splitext(path)
    moved_path = (
        f"{base}.{label}.{datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext or '.json'}"
    )
    os.replace(path, moved_path)
    return moved_path
