from stable_asr.roadmap import load_roadmap, roadmap_status, validate_roadmap


def test_roadmap_registry_validates_and_tracks_required_artifacts() -> None:
    roadmap = load_roadmap()
    validation = validate_roadmap(roadmap)
    report = roadmap_status(roadmap)

    assert validation.ok
    assert report.ok
    assert report.final_readiness is not None
    assert not report.final_scale_ready
    assert report.final_readiness.missing_required_inputs > 0
    assert "final_scale_ready: NO" in report.to_text()
    assert "Final-Scale Readiness" in report.to_markdown()
    assert roadmap["id"] == "stable_asr_roadmap_v0"
    assert len(report.milestones) >= 5
    assert not report.missing_required_artifacts
    assert any(milestone.id == "m2_data_reference_layer" for milestone in report.milestones)


def test_roadmap_status_surfaces_missing_required_artifacts(tmp_path) -> None:
    roadmap = {
        "id": "test_roadmap",
        "version": "0.0.0",
        "title": "Test Roadmap",
        "milestones": [
            {
                "id": "m0",
                "title": "M0",
                "status": "active",
                "objective": "check missing artifacts",
                "artifacts": [
                    {"path": "missing.txt", "required_now": True},
                    {"path": "planned.txt", "required_now": False},
                ],
                "commands": ["echo test"],
                "success_criteria": ["missing required artifact is reported"],
            }
        ],
    }

    report = roadmap_status(roadmap, repo_root=tmp_path)

    assert not report.ok
    assert report.final_readiness is None
    assert [artifact.path for artifact in report.missing_required_artifacts] == ["missing.txt"]
    assert "missing_required_artifacts: 1" in report.to_text()
    assert "Test Roadmap" in report.to_markdown()
