from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.experiments import run_paper_smoke


def test_paper_draft_generates_editable_markdown(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=7, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    output = tmp_path / "PAPER_DRAFT.md"

    generated = paper_draft(result.results_path, output, artifacts_dir=bundle.output_dir)
    draft = output.read_text(encoding="utf-8")

    assert generated == str(output)
    assert "# Stable-ASR:" in draft
    assert "## Abstract" in draft
    assert "## Related Work And Positioning" in draft
    assert "FunASR" in draft
    assert "Smart Turn" in draft
    assert "not as a replacement for mature ASR toolkits" in draft
    assert "## 6. Turn Latency And Deployment" in draft
    assert "| records | valid | languages |" in draft
    assert "| baseline | accuracy |" in draft
    assert "| baseline | category | count |" in draft
    assert "| baseline | avg_latency_ms |" in draft
    assert "| source | category | count |" in draft
    assert "paper-draft --results" in draft
    assert "![latency]" in draft
