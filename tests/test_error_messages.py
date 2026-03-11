import pytest

from module.error_messages import build_user_error_message, build_user_error_summary
from module.exceptions import ConfigError, ModelLoadError
from module.generation import generate_image


def test_build_user_error_message_for_missing_file_includes_guidance():
    message = build_user_error_message(
        FileNotFoundError("missing_model.safetensors"),
        action="画像生成",
    )

    assert "画像生成に失敗しました。" in message
    assert "必要なモデルまたはファイルが見つかりません。" in message
    assert "モデル一覧を更新するか" in message


def test_build_user_error_summary_for_config_error_is_user_friendly():
    summary = build_user_error_summary(ConfigError("bad yaml"))

    assert "設定内容に誤りがあります。" in summary
    assert "YAML の書式" in summary


def test_generate_image_raises_model_load_error_when_model_is_missing():
    with pytest.raises(ModelLoadError):
        generate_image("missing_model.safetensors")
