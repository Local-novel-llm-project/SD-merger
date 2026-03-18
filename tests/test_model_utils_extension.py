import sys
import types

import pytest

import extensions.model_utils as model_utils


def test_post_merge_utils_propagates_failures(monkeypatch):
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda path, device="cpu": {}
    fake_safetensors_torch.save_file = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)
    monkeypatch.setattr(
        model_utils,
        "_load_existing_metadata",
        lambda path: (_ for _ in ()).throw(RuntimeError("metadata failed")),
    )

    with pytest.raises(RuntimeError, match="metadata failed"):
        model_utils.post_merge_utils(
            {"custom_metadata": {"author": "tester"}},
            "merged_model.safetensors",
        )
