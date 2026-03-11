from module.extension_manager import register_pre_config_hook

CALLS = 0


def hook(config):
    return config


def setup():
    global CALLS
    CALLS += 1
    register_pre_config_hook(hook)
