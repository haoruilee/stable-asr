from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.completion import completion_audit
from stable_asr.paper.experiments import run_paper_smoke


def test_completion_audit_maps_goal_to_real_gates(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=17, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = completion_audit(results_path=result.results_path, artifacts_dir=bundle.output_dir)
    payload = report.to_dict()
    markdown = report.to_markdown()
    requirements = {item["requirement"] for item in payload["items"]}

    assert not report.ok
    assert payload["objective"] == "完成路线图,形成优秀的平台和论文,提供有价值的仓库"
    assert {
        "roadmap",
        "stable_worldmodel_style_platform",
        "paper_smoke_bundle",
        "paper_structural_parity",
        "final_inputs",
        "external_reference_evidence",
        "final_assignment",
        "final_handoff",
        "final_release_ready",
    }.issubset(requirements)
    assert "Stable-ASR Completion Audit" in markdown
    assert "Prompt-To-Artifact Checklist" in markdown
    assert "data/librispeech/LibriSpeech/dev-clean" in markdown
    assert "reference-workqueue --audit-evidence" in markdown
