import os

from extensions.auto_generate import auto_generate_hook
import extensions.auto_generate as auto_generate


class _FakeImage:
    def __init__(self):
        self.saved_paths = []

    def save(self, path):
        self.saved_paths.append(path)


def test_auto_generate_hook_updates_history_with_output_path(monkeypatch):
    captured = {}
    fake_image = _FakeImage()

    monkeypatch.setattr(
        auto_generate,
        "auto_generate_config",
        {
            "enabled": True,
            "prompt": "prompt",
            "negative_prompt": "negative",
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 1,
        },
    )
    monkeypatch.setattr(auto_generate, "generate_image", lambda **kwargs: [fake_image])

    def fake_update_history_entry(output_name, update_dict):
        captured["output_name"] = output_name
        captured["update_dict"] = update_dict
        return True

    monkeypatch.setattr(auto_generate, "update_history_entry", fake_update_history_entry)

    output_path = os.path.join("models", "output", "nested", "model.safetensors")
    auto_generate_hook({}, output_path)

    assert captured["output_name"] == output_path
    assert captured["update_dict"]["preview_image"].endswith(".png")
    assert captured["update_dict"]["generated_images"] == [
        captured["update_dict"]["preview_image"]
    ]
    assert fake_image.saved_paths == captured["update_dict"]["generated_images"]
