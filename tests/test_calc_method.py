import pytest
import torch

from module import calc_method


@pytest.mark.parametrize(
    ("strategy_func", "expected"),
    [
        (calc_method.subtract_velocity, torch.tensor([-1.5, 0.0])),
        (calc_method.add_velocity, torch.tensor([2.5, 2.0])),
        (calc_method.multiply_velocity, torch.tensor([2.0, 2.0])),
        (calc_method.average_velocity, torch.tensor([1.25, 1.0])),
        (calc_method.mix_velocity, torch.tensor([2.5, 2.0])),
    ],
)
def test_calculation_strategies_apply_expected_tensor_math(strategy_func, expected):
    result = strategy_func.__wrapped__(
        torch.tensor([1.0, 2.0]),
        torch.tensor([4.0, 2.0]),
        velocity=0.5,
        key_patterns_json='["block"]',
        key="model.block.weight",
    )

    assert torch.allclose(result, expected)


def test_replace_strategies_return_original_when_key_does_not_match():
    left = torch.tensor([1.0, 2.0])
    right = torch.tensor([4.0, 5.0])

    left_result = calc_method.replace_left_velocity.__wrapped__(
        left,
        right,
        velocity=0.5,
        key_patterns_json='["missing"]',
        key="model.block.weight",
    )
    right_result = calc_method.replace_right_velocity.__wrapped__(
        left,
        right,
        velocity=0.5,
        key_patterns_json='["missing"]',
        key="model.block.weight",
    )

    assert torch.equal(left_result, left)
    assert torch.equal(right_result, right)


def test_get_calculation_strategy_requires_replace_side():
    with pytest.raises(ValueError):
        calc_method.get_calculation_strategy("replace")


def test_get_calculation_strategy_uses_extension_registry(monkeypatch):
    def custom_strategy():
        return "custom"

    monkeypatch.setattr(
        calc_method,
        "get_extension_strategies",
        lambda: {"custom": custom_strategy},
    )

    assert calc_method.get_calculation_strategy("custom") is custom_strategy
