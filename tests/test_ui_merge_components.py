from ui.components import ab_test, mbw_each


def test_build_ab_merge_config_omits_left_right_velocity_when_blank(monkeypatch):
    monkeypatch.setattr(ab_test, "get_model_path", lambda name: f"/models/{name}")

    config = ab_test._build_ab_merge_config(
        "ModelA",
        "ModelB",
        "mix",
        0.5,
        "",
    )

    assert config["target_model"] == "/models/ModelA"
    assert config["models"][0]["left"] == "/models/ModelA"
    assert config["models"][0]["right"] == "/models/ModelB"
    assert config["models"][0]["velocity"] == 0.5
    assert "left_right_velocity" not in config["models"][0]


def test_build_ab_merge_config_includes_left_right_velocity(monkeypatch):
    monkeypatch.setattr(ab_test, "get_model_path", lambda name: f"/models/{name}")

    config = ab_test._build_ab_merge_config(
        "ModelA",
        "ModelB",
        "subtraction",
        0.25,
        "0.8",
    )

    assert config["models"][0]["left_right_velocity"] == 0.8


def test_build_mbw_each_task_config_omits_left_right_velocity_when_not_enabled(monkeypatch):
    monkeypatch.setattr(mbw_each, "get_model_path", lambda name: f"/models/{name}")

    config = mbw_each._build_mbw_each_task_config(
        "ModelA",
        "ModelB",
        None,
        "mbw_each",
        "mix",
        0.5,
        "1,1,1",
        "0,0,0",
        "",
        False,
        "out.safetensors",
    )

    assert config["target_model"] == "/models/ModelA"
    assert config["models"][0]["velocity"] == 0.5
    assert "left_right_velocity" not in config["models"][0]
    assert "output_name" not in config


def test_build_mbw_each_task_config_includes_optional_left_right_velocity(monkeypatch):
    monkeypatch.setattr(mbw_each, "get_model_path", lambda name: f"/models/{name}")

    config = mbw_each._build_mbw_each_task_config(
        "ModelA",
        "ModelB",
        "ModelC",
        "mbw_each",
        "addition",
        0.25,
        "1,1,1",
        "0,0,0",
        "0.65",
        True,
        "mbw.safetensors",
    )

    assert config["target_model"] == "/models/ModelC"
    assert config["models"][0]["velocity"] == 0.25
    assert config["models"][0]["left_right_velocity"] == 0.65
    assert config["output_name"] == "mbw.safetensors"
