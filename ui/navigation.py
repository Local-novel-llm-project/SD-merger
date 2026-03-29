from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationItem:
    label: str
    route: str
    title: str
    description: str


NAVIGATION_ITEMS: tuple[NavigationItem, ...] = (
    NavigationItem(
        label="Merge",
        route="/",
        title="SD-merger",
        description="モデルを選択してマージ設定を組み立て、キューへ投入します。",
    ),
    NavigationItem(
        label="Queue",
        route="/queue",
        title="Queue",
        description="バックグラウンドで進行中のタスクを確認し、操作します。",
    ),
    NavigationItem(
        label="History",
        route="/history",
        title="History",
        description="保存済みのレシピ履歴を確認し、再実行につなげます。",
    ),
    NavigationItem(
        label="Arthemy Tuner",
        route="/tune",
        title="Arthemy Tuner",
        description="主要パラメータを編集し、チューニングジョブを投入します。",
    ),
)


def get_navigation_items() -> tuple[NavigationItem, ...]:
    return NAVIGATION_ITEMS
