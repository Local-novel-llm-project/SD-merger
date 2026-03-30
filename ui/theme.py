from __future__ import annotations

import reflex as rx
from typing import Any, cast


THEME_TOKENS = {
    "font_family": "'IBM Plex Sans', 'Noto Sans JP', sans-serif",
    "page_background": (
        "linear-gradient(180deg, #fffaf0 0%, #fff3df 18%, #f8f5ef 58%, #f5f7f8 100%)"
    ),
    "sidebar_background": "rgba(255, 250, 240, 0.9)",
    "panel_background": "rgba(255, 255, 255, 0.92)",
    "panel_alt_background": "rgba(255, 249, 240, 0.96)",
    "panel_border": "rgba(180, 138, 74, 0.2)",
    "panel_shadow": "0 18px 45px rgba(94, 62, 30, 0.08)",
    "text_strong": "#312312",
    "text_muted": "#6a5b4d",
    "text_soft": "#8b7a68",
    "accent": "#b7791f",
    "accent_soft": "#fff1cf",
    "accent_border": "#d6a04a",
    "success_background": "#eefbf3",
    "success_border": "#16a34a",
    "success_text": "#166534",
    "warning_background": "#fff4e5",
    "warning_border": "#ea580c",
    "warning_text": "#9a3412",
    "error_background": "#fff1f2",
    "error_border": "#e11d48",
    "error_text": "#9f1239",
    "info_background": "#fff7e6",
    "info_border": "#f59e0b",
    "info_text": "#92400e",
    "status_running_background": "#e8f3ff",
    "status_running_border": "#2563eb",
    "status_running_text": "#1d4ed8",
    "status_pending_background": "#fff7e6",
    "status_pending_border": "#f59e0b",
    "status_pending_text": "#92400e",
    "status_completed_background": "#eefbf3",
    "status_completed_border": "#16a34a",
    "status_completed_text": "#166534",
    "status_error_background": "#fff1f2",
    "status_error_border": "#e11d48",
    "status_error_text": "#9f1239",
    "radius": "20px",
    "radius_small": "14px",
    "content_width": "1360px",
}


APP_THEME = {
    "appearance": "light",
    "accent_color": "amber",
    "gray_color": "sand",
    "radius": "large",
    "scaling": "105%",
    "has_background": True,
}


def create_theme() -> rx.Component:
    return rx.theme(**cast(dict[str, Any], APP_THEME))
