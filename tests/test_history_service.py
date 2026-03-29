import ui.services.history_service as history_service


def test_build_history_rows_handles_target_only_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "load_history",
        lambda: [
            {
                "date": "2026-03-16 00:00:00",
                "output_name": "arthemy_tuned.safetensors",
                "status": "Success",
                "config": {
                    "target_model": "models/base.safetensors",
                    "models": [],
                },
            }
        ],
    )

    rows = history_service.build_history_rows()

    assert len(rows) == 1
    assert rows[0]["strategy"] == "target-only"
    assert rows[0]["model_a"] == "models/base.safetensors"
    assert rows[0]["output_name"] == "arthemy_tuned.safetensors"


def test_build_history_rows_exposes_velocity_and_lrv(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged.safetensors",
                "status": "Success",
                "config": {
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                            "left_right_velocity": 0.75,
                        }
                    ],
                },
            }
        ],
    )

    rows = history_service.build_history_rows()

    assert rows[0]["velocity"] == "0.4"
    assert rows[0]["left_right_velocity"] == "0.75"


def test_build_history_rows_resolves_auto_lrv_for_ab_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_ab.safetensors",
                "status": "Success",
                "config": {
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                        }
                    ],
                },
            }
        ],
    )

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "0.4"


def test_build_history_rows_resolves_auto_lrv_for_legacy_ab_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_legacy_ab.safetensors",
                "status": "Success",
                "config": {
                    "target_model": "models/a.safetensors",
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                        }
                    ],
                },
            }
        ],
    )

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "0.4"


def test_build_history_rows_resolves_auto_lrv_for_target_entries(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_target.safetensors",
                "status": "Success",
                "config": {
                    "target_model": "models/base.safetensors",
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                        }
                    ],
                },
            }
        ],
    )

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "1.0"


def test_build_history_rows_supports_lr_alias(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "load_history",
        lambda: [
            {
                "date": "2026-03-23 00:00:00",
                "output_name": "merged_alias.safetensors",
                "status": "Success",
                "config": {
                    "models": [
                        {
                            "left": "models/a.safetensors",
                            "right": "models/b.safetensors",
                            "strategy": "mix",
                            "velocity": 0.4,
                            "lr": 0.9,
                        }
                    ],
                },
            }
        ],
    )

    rows = history_service.build_history_rows()

    assert rows[0]["left_right_velocity"] == "0.9"
