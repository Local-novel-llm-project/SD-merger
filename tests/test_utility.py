from pathlib import Path

from module import utility


class _FixedDateTime:
    @classmethod
    def now(cls):
        class _Now:
            def strftime(self, fmt: str) -> str:
                assert fmt == "%Y%m%d%H%M%S"
                return "20260311123045"

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

    assert filename == "abyoramix_flaani_20260311123045.safetensors"


def test_load_yaml_config_reads_yaml_file():
    config_path = Path("tests") / "fixtures" / "sample_test_config.yaml"
    loaded = utility.load_yaml_config(str(config_path))

    assert loaded["models"][0]["left"] == "model_a.safetensors"
    assert loaded["models"][0]["velocity"] == 0.5
