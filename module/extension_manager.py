import os
import sys
import importlib
from typing import Callable, Dict, List, Any, Optional

from module.logging_config import logger
from module.exceptions import ExtensionError

# マージ戦略名の登録ディクショナリ
# key: strategy_name, value: callable (sd_mecha merge_method)
_EXTENSION_STRATEGIES: Dict[str, Callable] = {}
_EXTENSION_TARGET_STRATEGIES: Dict[str, Callable] = {}

# イベントフックの登録リスト
_HOOKS_PRE_CONFIG: List[Callable] = []
_HOOKS_PRE_MERGE: List[Callable] = []
_HOOKS_POST_MERGE: List[Callable] = []
_LOADED_EXTENSION_MODULES: set[str] = set()


def register_strategy(name: str, func: Callable):
    """拡張機能から新しい計算戦略を登録する。"""
    if name in _EXTENSION_STRATEGIES:
        logger.warning(f"拡張機能戦略 '{name}' が上書きされました。")
    _EXTENSION_STRATEGIES[name] = func
    logger.info(f"拡張機能から戦略 '{name}' が登録されました。")


def register_target_strategy(name: str, func: Callable):
    """拡張機能から新しいターゲット計算戦略を登録する。"""
    if name in _EXTENSION_TARGET_STRATEGIES:
        logger.warning(f"拡張機能ターゲット戦略 '{name}' が上書きされました。")
    _EXTENSION_TARGET_STRATEGIES[name] = func
    logger.info(f"拡張機能からターゲット戦略 '{name}' が登録されました。")


def register_pre_config_hook(func: Callable):
    """設定読み込み直後（YAMLパース直後）に実行されるフックを登録する。"""
    if func not in _HOOKS_PRE_CONFIG:
        _HOOKS_PRE_CONFIG.append(func)


def register_pre_merge_hook(func: Callable):
    """マージ（レシピ結合）前に実行されるフックを登録する。"""
    if func not in _HOOKS_PRE_MERGE:
        _HOOKS_PRE_MERGE.append(func)


def register_post_merge_hook(func: Callable):
    """マージ実行後（ファイル出力後等）に実行されるフックを登録する。"""
    if func not in _HOOKS_POST_MERGE:
        _HOOKS_POST_MERGE.append(func)


def get_extension_strategies() -> Dict[str, Callable]:
    """登録済みの拡張機能戦略一覧を取得する。"""
    return _EXTENSION_STRATEGIES


def get_extension_target_strategies() -> Dict[str, Callable]:
    """登録済みの拡張機能ターゲット戦略一覧を取得する。"""
    return _EXTENSION_TARGET_STRATEGIES


def run_pre_config_hooks(config: dict) -> dict:
    """設定読み込み後のフックを実行し、構成内容を動的に変更する。"""
    for hook in _HOOKS_PRE_CONFIG:
        try:
            config = hook(config)
        except Exception as e:
            logger.error(f"pre_config フック実行中にエラーが発生しました: {e}")
            raise ExtensionError(
                f"pre_config フック実行中のエラー: {e}", original_error=e
            )
    return config


def run_pre_merge_hooks(config: dict, recipe: Any) -> Any:
    """レシピ結合前のフックを実行する。
    フックは recipe を受け取り、改変した recipe を返すことができる。
    """
    for hook in _HOOKS_PRE_MERGE:
        try:
            recipe = hook(config, recipe)
        except Exception as e:
            logger.error(f"pre_merge フック実行中にエラーが発生しました: {e}")
            raise ExtensionError(
                f"pre_merge フック実行中のエラー: {e}", original_error=e
            )
    return recipe


def run_post_merge_hooks(config: dict, output_path: str):
    """マージ終了後のフックを実行する。"""
    for hook in _HOOKS_POST_MERGE:
        try:
            hook(config, output_path)
        except Exception as e:
            logger.error(f"post_merge フック実行中にエラーが発生しました: {e}")
            raise ExtensionError(
                f"post_merge フック実行中のエラー: {e}", original_error=e
            )


def load_extensions(extensions_dir: Optional[str] = None):
    """
    指定されたディレクトリ内の拡張機能を動的に読み込む。
    各拡張機能ディレクトリの `__init__.py` (またはメインモジュール) にある
    `setup()` 関数が存在すればそれを呼び出す。
    """
    if extensions_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        extensions_dir = os.path.join(project_root, "extensions")

    if not os.path.exists(extensions_dir):
        logger.info(
            f"拡張機能ディレクトリ '{extensions_dir}' が見つかりませんでした。作成します。"
        )
        os.makedirs(extensions_dir, exist_ok=True)
        return

    # extensions フォルダを sys.path に追加してインポート可能にする
    if extensions_dir not in sys.path:
        sys.path.insert(0, extensions_dir)

    loaded_extensions = []

    for item in os.listdir(extensions_dir):
        item_path = os.path.join(extensions_dir, item)
        # フォルダであり、__init__.py が存在する場合のみ拡張として扱う
        if os.path.isdir(item_path) and os.path.exists(
            os.path.join(item_path, "__init__.py")
        ):
            try:
                # 拡張機能モジュールをインポート
                module = importlib.import_module(item)

                # setup() 関数があれば実行
                if hasattr(module, "setup"):
                    if item in _LOADED_EXTENSION_MODULES:
                        logger.info(f"拡張機能 '{item}' はすでに読み込み済みです。")
                        loaded_extensions.append(item)
                        continue
                    module.setup()
                    _LOADED_EXTENSION_MODULES.add(item)
                    loaded_extensions.append(item)
                else:
                    logger.warning(
                        f"拡張機能 '{item}' に setup() 関数が見つかりません。"
                    )

            except Exception as e:
                logger.error(
                    f"拡張機能 '{item}' の読み込み中にエラーが発生しました: {e}"
                )

    if loaded_extensions:
        logger.info(f"読み込まれた拡張機能: {', '.join(loaded_extensions)}")
    else:
        logger.info("拡張機能は読み込まれませんでした。")
