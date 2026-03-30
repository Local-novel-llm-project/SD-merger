from __future__ import annotations

import asyncio
from typing import Any

import reflex as rx

from module.error_messages import build_user_error_message
from ui.services.app_boot import ensure_app_ready
from ui.services.model_service import list_models

STATUS_VARIANTS = ("info", "success", "warning", "error")


class BasePageState(rx.State):
    status_message: str = ""
    status_variant: str = "info"
    busy: bool = False
    busy_message: str = ""

    available_models: list[str] = []

    polling_enabled: bool = False
    polling_generation: int = 0
    poll_interval_seconds: int = 2
    poll_route: str = ""

    def ensure_ready(self) -> None:
        ensure_app_ready()

    def set_status(self, message: str, variant: str = "info") -> None:
        self.status_message = message
        self.status_variant = variant if variant in STATUS_VARIANTS else "info"

    def clear_status(self) -> None:
        self.status_message = ""
        self.status_variant = "info"

    def set_info(self, message: str) -> None:
        self.set_status(message, "info")

    def set_success(self, message: str) -> None:
        self.set_status(message, "success")

    def set_warning(self, message: str) -> None:
        self.set_status(message, "warning")

    def set_error(self, exc: Exception, *, action: str) -> None:
        self.set_status(build_user_error_message(exc, action=action), "error")

    def begin_busy(self, message: str = "") -> None:
        self.busy = True
        self.busy_message = message

    def end_busy(self, message: str | None = None, variant: str = "success") -> None:
        self.busy = False
        self.busy_message = ""
        if message:
            self.set_status(message, variant)

    def fail_busy(self, exc: Exception, *, action: str) -> None:
        self.busy = False
        self.busy_message = ""
        self.set_error(exc, action=action)

    def _reconcile_model_selection(self) -> None:
        return None

    def refresh_models(
        self,
        *,
        action_name: str = "モデル一覧更新",
        success_message: str | None = None,
    ) -> None:
        self.begin_busy(action_name)
        try:
            self.available_models = list_models()
            self._reconcile_model_selection()
            resolved_message = success_message or f"Loaded {len(self.available_models)} models."
            self.end_busy(resolved_message, variant="info")
        except Exception as exc:
            self.available_models = []
            self.fail_busy(exc, action=action_name)

    def start_polling(self):
        self.polling_enabled = True
        self.polling_generation += 1
        return self.__class__.poll_state(self.polling_generation)

    def stop_polling(self) -> None:
        self.polling_enabled = False

    def _poll_should_stop(self, generation: int, current_path: str) -> bool:
        if not self.polling_enabled or self.polling_generation != generation:
            return True
        if self.poll_route and current_path != self.poll_route:
            return True
        return False

    def _poll_tick(self) -> None:
        return None

    @rx.event(background=True)
    async def poll_state(self, generation: int) -> None:
        interval_seconds = max(1, int(self.poll_interval_seconds))
        while True:
            async with self:
                current_path = getattr(self.router.page, "path", "")
                if self._poll_should_stop(generation, current_path):
                    self.polling_enabled = False
                    break
                self._poll_tick()
            await asyncio.sleep(interval_seconds)

    async def read_uploaded_text(
        self,
        files: list[rx.UploadFile],
        *,
        upload_id: str,
        empty_message: str,
        encoding: str = "utf-8",
    ) -> tuple[str | None, str | None, Any]:
        if not files:
            self.set_status(empty_message, "error")
            return None, None, None

        upload = files[0]
        contents = await upload.read()
        return contents.decode(encoding), upload.filename, rx.clear_selected_files(upload_id)

    def build_download(self, data: str, filename: str, *, message: str):
        self.set_info(message)
        return rx.download(data=data, filename=filename)
