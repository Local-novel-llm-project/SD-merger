import pytest
import torch

from module.generation import (
    _find_meta_tensor_names,
    _materialize_meta_module,
    _prepare_pipeline_for_device,
)


class _FakeMetaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, device="meta"))
        self.register_buffer("buffer", torch.empty(1, device="meta"))
        self.all_tied_weights_keys = {}
        self.move_calls = []
        self.initialize_calls = []

    def get_parameter_or_buffer(self, key: str):
        return getattr(self, key)

    def _move_missing_keys_from_meta_to_device(
        self,
        missing_keys,
        device_map,
        device_mesh,
        hf_quantizer,
    ):
        self.move_calls.append((set(missing_keys), device_map, device_mesh, hf_quantizer))
        target_device = next(iter(device_map.values()))
        self.weight = torch.nn.Parameter(torch.zeros(2, device=target_device))
        self.buffer = torch.zeros(1, device=target_device)

    def _initialize_missing_keys(self, is_quantized: bool):
        self.initialize_calls.append(is_quantized)


class _FakeUnsupportedMetaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, device="meta"))


class _FakePipeline:
    def __init__(self, module: torch.nn.Module):
        self.components = {
            "text_encoder": module,
            "tokenizer": object(),
        }


def test_materialize_meta_module_initializes_transformers_style_module():
    module = _FakeMetaModule()

    assert _find_meta_tensor_names(module) == {"buffer", "weight"}

    _materialize_meta_module(module, "cpu")

    assert _find_meta_tensor_names(module) == set()
    assert module.move_calls == [({"buffer", "weight"}, {"": torch.device("cpu")}, None, None)]
    assert module.initialize_calls == [False]
    assert module.weight.device.type == "cpu"
    assert module.buffer.device.type == "cpu"


def test_materialize_meta_module_raises_for_unsupported_module():
    module = _FakeUnsupportedMetaModule()

    with pytest.raises(RuntimeError, match="安全に実体化する手段が見つかりませんでした"):
        _materialize_meta_module(module, "cpu")


def test_prepare_pipeline_for_device_materializes_only_torch_modules():
    module = _FakeMetaModule()
    pipe = _FakePipeline(module)

    _prepare_pipeline_for_device(pipe, "cpu")

    assert _find_meta_tensor_names(module) == set()
    assert len(module.move_calls) == 1
