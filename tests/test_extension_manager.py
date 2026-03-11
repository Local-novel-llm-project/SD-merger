import sys
from pathlib import Path

from module import extension_manager


def _reset_extension_manager_state():
    extension_manager._EXTENSION_STRATEGIES.clear()
    extension_manager._EXTENSION_TARGET_STRATEGIES.clear()
    extension_manager._HOOKS_PRE_CONFIG.clear()
    extension_manager._HOOKS_PRE_MERGE.clear()
    extension_manager._HOOKS_POST_MERGE.clear()
    extension_manager._LOADED_EXTENSION_MODULES.clear()


def test_register_pre_config_hook_ignores_duplicates():
    _reset_extension_manager_state()

    def hook(config):
        return config

    extension_manager.register_pre_config_hook(hook)
    extension_manager.register_pre_config_hook(hook)

    assert extension_manager._HOOKS_PRE_CONFIG == [hook]
    _reset_extension_manager_state()


def test_load_extensions_only_runs_setup_once():
    _reset_extension_manager_state()

    extensions_dir = Path(__file__).parent / "fixtures" / "extensions"
    sys.modules.pop("demo_ext", None)

    try:
        extension_manager.load_extensions(str(extensions_dir))
        extension_manager.load_extensions(str(extensions_dir))

        loaded_module = sys.modules["demo_ext"]
        assert loaded_module.CALLS == 1
        assert len(extension_manager._HOOKS_PRE_CONFIG) == 1
    finally:
        sys.modules.pop("demo_ext", None)
        if str(extensions_dir) in sys.path:
            sys.path.remove(str(extensions_dir))
        _reset_extension_manager_state()
