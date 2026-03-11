import pytest
import torch

from module import calc_target


@pytest.mark.parametrize(
    ("strategy_func", "expected"),
    [
        (calc_target.target_addition, torch.tensor([2.0, 2.5])),
        (calc_target.target_subtraction, torch.tensor([0.0, 1.5])),
        (calc_target.target_multiplication, torch.tensor([1.0, 1.0])),
        (calc_target.target_mix, torch.tensor([1.5, 1.5])),
    ],
)
def test_target_strategies_apply_expected_tensor_math(strategy_func, expected):
    result = strategy_func.__wrapped__(
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 1.0]),
        velocity=0.5,
        key_patterns_json='["block"]',
        key="model.block.weight",
    )

    assert torch.allclose(result, expected)


def test_target_strategy_returns_target_when_key_does_not_match():
    target = torch.tensor([1.0, 2.0])
    diff = torch.tensor([3.0, 4.0])

    result = calc_target.target_addition.__wrapped__(
        target,
        diff,
        velocity=0.5,
        key_patterns_json='["missing"]',
        key="model.block.weight",
    )

    assert torch.equal(result, target)


def test_normalize_std_mean_matches_original_statistics():
    target = torch.tensor([1.0, 2.0, 3.0])
    merged = torch.tensor([10.0, 20.0, 30.0])

    normalized = calc_target.normalize_std_mean.__wrapped__(target, merged)

    target_std, target_mean = torch.std_mean(target.float())
    normalized_std, normalized_mean = torch.std_mean(normalized.float())

    assert torch.allclose(normalized_mean, target_mean)
    assert torch.allclose(normalized_std, target_std)


def test_target_and_normalization_strategy_getters_support_extensions(monkeypatch):
    def custom_target():
        return "target"

    monkeypatch.setattr(
        "module.extension_manager.get_extension_target_strategies",
        lambda: {"custom_target": custom_target},
    )

    assert calc_target.get_target_calculation_strategy("custom_target") is custom_target
    assert calc_target.get_normalization_calculation_strategy("none") is None

    with pytest.raises(ValueError):
        calc_target.get_normalization_calculation_strategy("unknown")
