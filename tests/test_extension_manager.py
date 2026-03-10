import sys

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


def test_load_extensions_only_runs_setup_once(tmp_path):
    _reset_extension_manager_state()

    extensions_dir = tmp_path / "extensions"
    module_dir = extensions_dir / "demo_ext"
    module_dir.mkdir(parents=True)
    module_dir.joinpath("__init__.py").write_text(
        "from module.extension_manager import register_pre_config_hook\n"
        "CALLS = 0\n"
        "def hook(config):\n"
        "    return config\n"
        "def setup():\n"
        "    global CALLS\n"
        "    CALLS += 1\n"
        "    register_pre_config_hook(hook)\n",
        encoding="utf-8",
    )

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
