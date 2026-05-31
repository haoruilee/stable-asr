"""Paper experiment helpers."""

from stable_asr.paper.artifacts import PaperArtifactBundle, paper_artifact_bundle
from stable_asr.paper.audit import (
    PaperAuditCheck,
    PaperAuditReport,
    PaperReleaseAuditCheck,
    PaperReleaseAuditReport,
    audit_paper_artifacts,
    audit_paper_release,
)
from stable_asr.paper.cards import dataset_card, experiment_card
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.evidence import FinalEvidenceMatrixReport, final_evidence_matrix
from stable_asr.paper.experiments import PaperRunResult, run_paper_smoke
from stable_asr.paper.figures import PAPER_FIGURES, paper_figure
from stable_asr.paper.final_config import FinalRunActionPlan, build_final_run_action_plan
from stable_asr.paper.final_results import FinalResultsAssemblyReport, assemble_final_paper_results
from stable_asr.paper.integrity import (
    ArtifactDigest,
    ArtifactIntegrityReport,
    artifact_integrity_manifest,
    load_artifact_integrity,
    verify_artifact_integrity,
    write_artifact_integrity,
)
from stable_asr.paper.latex import paper_latex
from stable_asr.paper.leaderboard import export_leaderboard, validate_leaderboard_jsonl
from stable_asr.paper.release_smoke import PaperReleaseSmokeResult, run_paper_release_smoke
from stable_asr.paper.suites import (
    DEFAULT_BENCHMARK_SUITE,
    DEFAULT_SUITE_ID,
    BenchmarkSuiteCoverage,
    BenchmarkSuiteValidation,
    audit_benchmark_suite_coverage,
    benchmark_suite_markdown,
    load_benchmark_suite,
    validate_benchmark_suite,
    write_benchmark_suite_json,
)
from stable_asr.paper.tables import PAPER_TABLES, paper_table

__all__ = [
    "PAPER_FIGURES",
    "PAPER_TABLES",
    "DEFAULT_BENCHMARK_SUITE",
    "DEFAULT_SUITE_ID",
    "BenchmarkSuiteCoverage",
    "BenchmarkSuiteValidation",
    "PaperArtifactBundle",
    "ArtifactDigest",
    "ArtifactIntegrityReport",
    "PaperAuditCheck",
    "PaperAuditReport",
    "PaperReleaseAuditCheck",
    "PaperReleaseAuditReport",
    "PaperReleaseSmokeResult",
    "PaperRunResult",
    "FinalResultsAssemblyReport",
    "FinalRunActionPlan",
    "FinalEvidenceMatrixReport",
    "assemble_final_paper_results",
    "artifact_integrity_manifest",
    "audit_paper_artifacts",
    "audit_paper_release",
    "build_final_run_action_plan",
    "dataset_card",
    "experiment_card",
    "export_leaderboard",
    "audit_benchmark_suite_coverage",
    "benchmark_suite_markdown",
    "load_benchmark_suite",
    "load_artifact_integrity",
    "paper_artifact_bundle",
    "paper_draft",
    "final_evidence_matrix",
    "paper_figure",
    "paper_latex",
    "paper_table",
    "run_paper_release_smoke",
    "run_paper_smoke",
    "validate_benchmark_suite",
    "validate_leaderboard_jsonl",
    "verify_artifact_integrity",
    "write_artifact_integrity",
    "write_benchmark_suite_json",
]
