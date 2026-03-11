from __future__ import annotations

from typing import Tuple

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - pydantic is an installed dependency in normal runs
    ValidationError = None  # type: ignore[assignment]

from module.exceptions import (
    ConfigError,
    ExtensionError,
    GenerationError,
    MergeError,
    ModelLoadError,
    SDMergerError,
)


def _unwrap_exception(exc: Exception) -> Exception:
    current: Exception = exc
    seen: set[int] = set()

    while isinstance(current, SDMergerError) and current.original_error is not None:
        next_error = current.original_error
        if id(next_error) in seen:
            break
        seen.add(id(next_error))
        current = next_error

    return current


def _normalise_detail(detail: str) -> str:
    cleaned = " ".join(detail.split())
    if len(cleaned) > 180:
        cleaned = cleaned[:177] + "..."
    return cleaned


def _classify_exception(exc: Exception) -> Tuple[str, str]:
    root = _unwrap_exception(exc)

    if isinstance(exc, ConfigError) or (
        ValidationError is not None and isinstance(root, ValidationError)
    ):
        return (
            "設定内容に誤りがあります。",
            "必須項目、数値の範囲、YAML の書式を確認してください。",
        )

    if isinstance(exc, ExtensionError):
        return (
            "拡張機能の処理に失敗しました。",
            "拡張設定を見直すか、一時的に無効化して再試行してください。",
        )

    if isinstance(exc, MergeError):
        return (
            "モデルのマージに失敗しました。",
            "モデルの組み合わせ、戦略、重み設定を見直してください。",
        )

    if isinstance(exc, GenerationError):
        return (
            "画像生成に失敗しました。",
            "モデル形式、VRAM 使用量、生成パラメータを確認してください。",
        )

    if isinstance(exc, ModelLoadError) or isinstance(root, FileNotFoundError):
        return (
            "必要なモデルまたはファイルが見つかりません。",
            "モデル一覧を更新するか、指定したパスと出力先を確認してください。",
        )

    if isinstance(root, PermissionError):
        return (
            "ファイルまたはフォルダにアクセスできません。",
            "別のアプリで開いていないか、書き込み権限があるか確認してください。",
        )

    if isinstance(root, ValueError):
        return (
            "入力値の形式が正しくありません。",
            "入力形式と数値の範囲を確認してから再試行してください。",
        )

    return (
        "予期しないエラーが発生しました。",
        "設定を見直して再試行し、解決しない場合はログを確認してください。",
    )


def build_user_message(
    action: str,
    reason: str,
    guidance: str,
    *,
    detail: str | None = None,
) -> str:
    lines = [
        f"{action}に失敗しました。",
        f"原因: {reason}",
        f"対処: {guidance}",
    ]
    if detail:
        lines.append(f"詳細: {_normalise_detail(detail)}")
    return "\n".join(lines)


def build_user_error_message(exc: Exception, action: str = "処理") -> str:
    reason, guidance = _classify_exception(exc)
    detail = _normalise_detail(str(_unwrap_exception(exc)))
    if detail == reason:
        detail = ""
    return build_user_message(action, reason, guidance, detail=detail or None)


def build_user_error_summary(exc: Exception) -> str:
    reason, guidance = _classify_exception(exc)
    return f"{reason} {guidance}"
