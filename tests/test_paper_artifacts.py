from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.figures import PAPER_FIGURES
from stable_asr.paper.tables import PAPER_TABLES


def test_paper_artifact_bundle_generates_tables_figures_and_index(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=5, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    assert Path(bundle.index_path).exists()
    assert Path(bundle.manifest_path).exists()
    assert set(bundle.tables) == set(PAPER_TABLES)
    assert set(bundle.figures) == set(PAPER_FIGURES)
    assert set(bundle.leaderboards) == {"jsonl", "csv"}
    assert set(bundle.benchmark_suite) == {"json", "markdown"}
    assert set(bundle.data_sources) == {"json", "markdown"}
    assert set(bundle.adapter_registry) == {"json", "markdown"}
    assert set(bundle.scenario_suite) == {"json", "markdown"}
    assert set(bundle.case_studies) == {"json", "markdown"}
    assert set(bundle.paper_parity) == {"json", "markdown"}
    assert set(bundle.final_experiments) == {"json", "markdown"}
    assert set(bundle.final_run_config) == {"json", "markdown"}
    assert set(bundle.final_run_file_audit) == {"json", "markdown"}
    assert set(bundle.paper_status) == {"json", "markdown"}
    assert set(bundle.claims) == {"json", "markdown"}
    assert "Stable-ASR Paper Artifact Index" in Path(bundle.index_path).read_text(encoding="utf-8")
    assert "rule_endpoint" in Path(bundle.tables["baselines"]).read_text(encoding="utf-8")
    assert "Stable-ASR Platform Architecture" in Path(bundle.figures["architecture"]).read_text(encoding="utf-8")
    assert "Baseline Macro F1" in Path(bundle.figures["baselines"]).read_text(encoding="utf-8")
    assert "turn_quality" in Path(bundle.leaderboards["jsonl"]).read_text(encoding="utf-8")
    assert "asr_transcript_conversion" in Path(bundle.benchmark_suite["markdown"]).read_text(encoding="utf-8")
    assert "synthetic_voiceworld" in Path(bundle.data_sources["markdown"]).read_text(encoding="utf-8")
    assert "command_streaming_asr" in Path(bundle.adapter_registry["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR VoiceWorld v0 Scenario Suite" in Path(bundle.scenario_suite["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Case Studies" in Path(bundle.case_studies["markdown"]).read_text(encoding="utf-8")
    assert "final-scale ready" in Path(bundle.paper_parity["markdown"]).read_text(encoding="utf-8")
    assert "real_data_layer_benchmark" in Path(bundle.final_experiments["markdown"]).read_text(encoding="utf-8")
    assert "librispeech_dev_clean" in Path(bundle.final_run_config["markdown"]).read_text(encoding="utf-8")
    assert "Final Run File Audit" in Path(bundle.final_run_file_audit["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Paper Status" in Path(bundle.paper_status["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Claim Evidence Matrix" in Path(bundle.claims["markdown"]).read_text(encoding="utf-8")
