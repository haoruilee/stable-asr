"""Paper experiment helpers."""

from stable_asr.paper.archive import (
    PaperArchiveReport,
    PaperArchiveVerificationReport,
    load_paper_archive_report,
    paper_artifact_archive,
    verify_paper_artifact_archive,
    write_paper_archive_report,
)
from stable_asr.paper.artifacts import PaperArtifactBundle, paper_artifact_bundle
from stable_asr.paper.audit import (
    PaperAuditCheck,
    PaperAuditReport,
    PaperReleaseAuditCheck,
    PaperReleaseAuditReport,
    audit_paper_artifacts,
    audit_paper_release,
)
from stable_asr.paper.acquisition_pack import (
    FinalAcquisitionPackReport,
    FinalAssignmentAuditReport,
    audit_acquisition_assignments,
    build_final_acquisition_pack,
)
from stable_asr.paper.adapter_pack import AdapterPackReport, build_adapter_pack
from stable_asr.paper.benchmark_pack import BenchmarkPackReport, build_benchmark_pack
from stable_asr.paper.cards import dataset_card, experiment_card, model_card, model_card_payload, write_model_card_json
from stable_asr.paper.contributor_pack import ContributorPackReport, build_contributor_pack
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.evidence import FinalEvidenceMatrixReport, final_evidence_matrix
from stable_asr.paper.experiments import PaperRunResult, run_paper_smoke
from stable_asr.paper.figures import PAPER_FIGURES, paper_figure
from stable_asr.paper.final_config import FinalRunActionPlan, build_final_run_action_plan
from stable_asr.paper.final_inputs import (
    FinalInputCollectionReport,
    final_input_collection_report,
    load_final_input_collections,
    validate_final_input_collections,
)
from stable_asr.paper.final_pack import FinalPackReport, build_final_pack
from stable_asr.paper.final_results import FinalResultsAssemblyReport, assemble_final_paper_results
from stable_asr.paper.handoff import (
    FINAL_HANDOFF_VERSION,
    FinalHandoffAuditReport,
    audit_final_handoff,
    final_handoff_template,
)
from stable_asr.paper.integrity import (
    ArtifactDigest,
    ArtifactIntegrityReport,
    artifact_integrity_manifest,
    load_artifact_integrity,
    verify_artifact_integrity,
    write_artifact_integrity,
)
from stable_asr.paper.latex import paper_latex
from stable_asr.paper.leaderboard import (
    LeaderboardMergeReport,
    export_leaderboard,
    leaderboard_report,
    merge_leaderboard_jsonl,
    validate_leaderboard_jsonl,
)
from stable_asr.paper.provenance import (
    GitProvenance,
    PaperProvenanceReport,
    ProvenanceFile,
    paper_bundle_provenance,
    write_paper_provenance,
)
from stable_asr.paper.release_smoke import PaperReleaseSmokeResult, run_paper_release_smoke
from stable_asr.paper.platform_parity import (
    PlatformParityReport,
    audit_platform_parity,
    load_platform_parity,
    validate_platform_parity,
)
from stable_asr.paper.scenario_pack import ScenarioPackReport, build_scenario_pack
from stable_asr.paper.submissions import SubmissionIndexReport, index_submission_directory
from stable_asr.paper.suites import (
    DEFAULT_BENCHMARK_SUITE,
    DEFAULT_SUITE_ID,
    BenchmarkArtifactAudit,
    BenchmarkSuiteCoverage,
    BenchmarkSuiteValidation,
    audit_benchmark_required_artifacts,
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
    "BenchmarkArtifactAudit",
    "BenchmarkSuiteCoverage",
    "BenchmarkSuiteValidation",
    "AdapterPackReport",
    "BenchmarkPackReport",
    "ContributorPackReport",
    "PaperArtifactBundle",
    "PaperArchiveReport",
    "PaperArchiveVerificationReport",
    "ArtifactDigest",
    "ArtifactIntegrityReport",
    "PaperAuditCheck",
    "PaperAuditReport",
    "PaperReleaseAuditCheck",
    "PaperReleaseAuditReport",
    "PaperReleaseSmokeResult",
    "PlatformParityReport",
    "PaperRunResult",
    "FinalResultsAssemblyReport",
    "FinalRunActionPlan",
    "FinalAcquisitionPackReport",
    "FinalAssignmentAuditReport",
    "FinalInputCollectionReport",
    "FinalEvidenceMatrixReport",
    "FinalHandoffAuditReport",
    "FinalPackReport",
    "LeaderboardMergeReport",
    "SubmissionIndexReport",
    "ScenarioPackReport",
    "GitProvenance",
    "PaperProvenanceReport",
    "ProvenanceFile",
    "assemble_final_paper_results",
    "artifact_integrity_manifest",
    "audit_acquisition_assignments",
    "audit_final_handoff",
    "audit_paper_artifacts",
    "audit_paper_release",
    "audit_platform_parity",
    "build_final_run_action_plan",
    "build_adapter_pack",
    "build_benchmark_pack",
    "build_contributor_pack",
    "build_final_acquisition_pack",
    "build_final_pack",
    "build_scenario_pack",
    "dataset_card",
    "experiment_card",
    "model_card",
    "model_card_payload",
    "index_submission_directory",
    "export_leaderboard",
    "leaderboard_report",
    "merge_leaderboard_jsonl",
    "audit_benchmark_required_artifacts",
    "audit_benchmark_suite_coverage",
    "benchmark_suite_markdown",
    "load_benchmark_suite",
    "load_artifact_integrity",
    "load_paper_archive_report",
    "paper_bundle_provenance",
    "paper_artifact_archive",
    "paper_artifact_bundle",
    "paper_draft",
    "final_evidence_matrix",
    "final_handoff_template",
    "final_input_collection_report",
    "FINAL_HANDOFF_VERSION",
    "load_final_input_collections",
    "load_platform_parity",
    "paper_figure",
    "paper_latex",
    "paper_table",
    "run_paper_release_smoke",
    "run_paper_smoke",
    "validate_benchmark_suite",
    "validate_leaderboard_jsonl",
    "validate_final_input_collections",
    "validate_platform_parity",
    "verify_artifact_integrity",
    "verify_paper_artifact_archive",
    "write_artifact_integrity",
    "write_benchmark_suite_json",
    "write_model_card_json",
    "write_paper_archive_report",
    "write_paper_provenance",
]
