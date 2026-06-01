import json
from pathlib import Path

from stable_asr.models.registry import audit_model_registry_configs, load_model_registry


def test_model_registry_config_audit_accepts_trainable_configs() -> None:
    report = audit_model_registry_configs(load_model_registry(), repo_root=".")

    assert report.ok
    rows = {row.model_id: row for row in report.rows}
    assert set(rows) == {"nanoturn_pico", "nanoturn_nano"}
    assert rows["nanoturn_pico"].exists
    assert rows["nanoturn_nano"].config_path == "configs/nanoturn_nano.json"
    assert "Stable-ASR Model Config Audit" in report.to_markdown()


def test_model_registry_config_audit_rejects_missing_config(tmp_path: Path) -> None:
    registry = load_model_registry()
    for model in registry["models"]:
        if model["id"] == "nanoturn_nano":
            model["config_path"] = "configs/does_not_exist.json"

    report = audit_model_registry_configs(registry, repo_root=tmp_path)

    assert not report.ok
    assert "nanoturn_nano" in report.errors[0]
    assert "missing" in report.errors[0]


def test_model_registry_config_audit_rejects_mismatched_model_type(tmp_path: Path) -> None:
    config = tmp_path / "bad_nano.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "nanoturn_pico",
                "epochs": 1,
                "lr": 0.01,
                "seed": 0,
                "feature_source": "manifest_metadata_v0",
            }
        ),
        encoding="utf-8",
    )
    registry = load_model_registry()
    for model in registry["models"]:
        if model["id"] == "nanoturn_nano":
            model["config_path"] = str(config)

    report = audit_model_registry_configs(registry, repo_root=tmp_path)

    assert not report.ok
    assert "expected 'nanoturn_nano'" in report.errors[0]


def test_model_registry_config_audit_rejects_schema_error(tmp_path: Path) -> None:
    config = tmp_path / "bad_nano.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "nanoturn_nano",
                "epochs": 1,
                "lr": 0.01,
                "seed": 0,
                "feature_source": "unknown_features",
            }
        ),
        encoding="utf-8",
    )
    registry = load_model_registry()
    for model in registry["models"]:
        if model["id"] == "nanoturn_nano":
            model["config_path"] = str(config)

    report = audit_model_registry_configs(registry, repo_root=tmp_path)

    assert not report.ok
    assert "$.feature_source" in report.errors[0]
