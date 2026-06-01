from pathlib import Path

from stable_asr.paper.platform_parity import (
    audit_platform_parity,
    load_platform_parity,
    validate_platform_parity,
)


def test_platform_parity_registry_validates_and_audits_repo() -> None:
    registry = load_platform_parity()
    validation = validate_platform_parity(registry)
    report = audit_platform_parity(registry)

    assert validation.ok
    assert report.ok
    assert report.missing_count == 0
    assert {check.item_id for check in report.checks} >= {
        "installable_cli_surface",
        "data_format_registry",
        "scenario_environment_suite",
        "baseline_adapter_solver_zoo",
        "paper_release_pipeline",
        "reference_collections",
    }
    installable = next(check for check in report.checks if check.item_id == "installable_cli_surface")
    assert installable.required_commands >= 5


def test_platform_parity_markdown_mentions_stable_worldmodel_shape() -> None:
    report = audit_platform_parity(load_platform_parity())
    markdown = report.to_markdown()

    assert "Stable-ASR Stable-WorldModel Repository Parity" in markdown
    assert "source_reference" in markdown
    assert "data_format_registry" in markdown
    assert "paper_release_pipeline" in markdown
    assert "installable_cli_surface" in markdown


def test_platform_parity_surfaces_missing_marker() -> None:
    registry = load_platform_parity()
    registry["items"][0]["required_markers"].append(
        {"path": "README.md", "contains": ["this marker should not exist"]}
    )

    report = audit_platform_parity(registry)

    assert not report.ok
    assert "README.md:this marker should not exist" in report.to_text()


def test_platform_parity_resolves_platform_assets_from_empty_repo_root(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    report = audit_platform_parity(repo_root=empty_root)

    assert report.ok
