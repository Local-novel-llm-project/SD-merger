import importlib
import os
import sys

import torch


EXTENSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "extensions")
)
if EXTENSIONS_DIR not in sys.path:
    sys.path.insert(0, EXTENSIONS_DIR)

advanced_calc = importlib.import_module("advanced_calc")


def test_target_angle_velocity_zero_preserves_target():
    target = torch.tensor([1.0, 2.0])
    diff_l = torch.tensor([1.0, 0.0])
    diff_r = torch.tensor([1.0, 0.0])
    left = torch.tensor([3.0, 4.0])
    right = torch.tensor([5.0, 6.0])

    result = advanced_calc.target_angle.__wrapped__(
        target,
        diff_l,
        diff_r,
        left,
        right,
        velocity=0.0,
        key_patterns_json='["block"]',
        key="model.block.weight",
    )

    assert torch.equal(result, target)


def test_train_difference_uses_provided_diffs():
    result = advanced_calc.target_train_difference.__wrapped__(
        torch.tensor([10.0]),
        torch.tensor([2.0]),
        torch.tensor([4.0]),
        torch.tensor([100.0]),
        torch.tensor([-100.0]),
        velocity=0.5,
        key_patterns_json='["block"]',
        key="model.block.weight",
    )

    assert torch.allclose(result, torch.tensor([11.2]))
