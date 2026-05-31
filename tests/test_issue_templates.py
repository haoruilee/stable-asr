from pathlib import Path


ISSUE_TEMPLATE_DIR = Path(".github/ISSUE_TEMPLATE")


def test_github_issue_templates_cover_contribution_tracks() -> None:
    expected = {
        "config.yml",
        "final_data_acquisition.yml",
        "asr_adapter.yml",
        "voiceworld_scenario.yml",
        "benchmark_submission.yml",
    }

    assert expected.issubset({path.name for path in ISSUE_TEMPLATE_DIR.glob("*.yml")})

    final_data = (ISSUE_TEMPLATE_DIR / "final_data_acquisition.yml").read_text(encoding="utf-8")
    assert "stable-asr final-acquisition-pack" in final_data
    assert "I did not add placeholder data" in final_data

    adapter = (ISSUE_TEMPLATE_DIR / "asr_adapter.yml").read_text(encoding="utf-8")
    assert "stable-asr adapter-pack" in adapter
    assert "The upstream license or terms are linked" in adapter

    voiceworld = (ISSUE_TEMPLATE_DIR / "voiceworld_scenario.yml").read_text(encoding="utf-8")
    assert "stable-asr scenario-pack" in voiceworld
    assert "Factor annotations" in voiceworld

    submission = (ISSUE_TEMPLATE_DIR / "benchmark_submission.yml").read_text(encoding="utf-8")
    assert "stable-asr benchmark-pack" in submission
    assert "leaderboard-validate" in submission


def test_pull_request_template_requires_evidence_and_final_scale_context() -> None:
    template = Path(".github/PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

    assert "Commands run" in template
    assert "Data, License, And Provenance" in template
    assert "Final-Scale Impact" in template
    assert "final-acquisition-pack" in template
    assert "contributor-pack" in template
    assert "paper-release-smoke" in template
