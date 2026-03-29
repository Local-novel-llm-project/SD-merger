from __future__ import annotations

import threading

_BOOT_LOCK = threading.Lock()
_BOOTSTRAPPED = False


def ensure_app_ready() -> None:
    global _BOOTSTRAPPED

    if _BOOTSTRAPPED:
        return

    with _BOOT_LOCK:
        if _BOOTSTRAPPED:
            return

        from module.extension_manager import load_extensions
        from module.queue_manager import queue_manager

        load_extensions()
        queue_manager.start_worker()
        _BOOTSTRAPPED = True
