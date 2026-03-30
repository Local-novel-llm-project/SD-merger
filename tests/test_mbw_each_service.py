import pytest

from module.services.mbw_each import (
    build_curve_weights,
    build_key_pattern_rules,
    build_mbw_each_config,
    build_mbw_each_model_config,
    resolve_block_keywords,
)


def test_build_mbw_each_model_config_expands_key_patterns():
    model_config = build_mbw_each_model_config(
        {
            "left": "a",
            "right": "b",
            "strategy": "mbw_each",
            "mbw_a": ",".join(["1"] * 26),
            "mbw_b": ",".join(["0"] * 26),
        }
    )

    assert "mbw_a" not in model_config
    assert "mbw_b" not in model_config
    assert len(model_config["key_patterns"]) == 26
    assert model_config["key_patterns"]["block_0"]["pattern"] == "cond_stage_model"


def test_build_mbw_each_config_updates_only_matching_entries():
    config = build_mbw_each_config(
        {
            "models": [
                {"left": "a", "right": "b", "mbw_a": ",".join(["1"] * 20), "mbw_b": ",".join(["0"] * 20)},
                {"left": "c", "right": "d", "strategy": "mix"},
            ]
        }
    )

    assert "key_patterns" in config["models"][0]
    assert config["models"][1]["strategy"] == "mix"


def test_build_key_pattern_rules_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_key_pattern_rules([1.0], [0.0, 1.0])


def test_resolve_block_keywords_rejects_unknown_block_count():
    with pytest.raises(ValueError):
        resolve_block_keywords(10)


def test_build_curve_weights_formats_mbw_strings():
    result = build_curve_weights(block_count=4, start=0.0, end=1.0, profile="ease_in_out")

    assert result.profile == "ease_in_out"
    assert len(result.weights) == 4
    assert result.mbw_a.count(",") == 3
    assert result.mbw_b.count(",") == 3
