from pathlib import Path
from datetime import datetime

from module import utility


class _FixedDateTime:
    @classmethod
    def now(cls):
        class _Now:
            def strftime(self, fmt: str) -> str:
                assert fmt == "%Y%m%d%H%M%S%f"
                return "20260311123045123456"

        return _Now()


def test_normalize_model_path_appends_extension_once():
    assert utility._normalize_model_path("models/base_model") == (
        "models/base_model.safetensors"
    )
    assert utility._normalize_model_path("models/base_model.safetensors") == (
        "models/base_model.safetensors"
    )


def test_generate_filename_uses_initials_and_timestamp(monkeypatch):
    monkeypatch.setattr(utility, "datetime", _FixedDateTime)

    filename = utility.generate_filename(
        "abyss_orange_mix.safetensors",
        "flat2D_anime.safetensors",
    )

    assert filename == "abyoramix_flaani_20260311123045123456.safetensors"


def test_load_yaml_config_reads_yaml_file():
    config_path = Path("tests") / "fixtures" / "sample_test_config.yaml"
    loaded = utility.load_yaml_config(str(config_path))

    assert loaded["models"][0]["left"] == "model_a.safetensors"
    assert loaded["models"][0]["velocity"] == 0.5


def _make_runtime_dir(name: str) -> Path:
    root = Path(__file__).parent / ".runtime"
    root.mkdir(exist_ok=True)
    path = root / name
    path.mkdir(exist_ok=True)
    return path


def test_load_yaml_config_returns_empty_dict_for_empty_file():
    config_path = _make_runtime_dir("utility") / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert utility.load_yaml_config(str(config_path)) == {}


def test_generate_filename_uses_high_resolution_timestamp(monkeypatch):
    class _IncrementingDateTime:
        values = iter(
            [
                datetime(2026, 3, 16, 4, 40, 0, 123456),
                datetime(2026, 3, 16, 4, 40, 0, 123457),
            ]
        )

        @classmethod
        def now(cls):
            return next(cls.values)

    monkeypatch.setattr(utility, "datetime", _IncrementingDateTime)

    first = utility.generate_filename("model_a.safetensors", "model_b.safetensors")
    second = utility.generate_filename("model_a.safetensors", "model_b.safetensors")

    assert first != second
