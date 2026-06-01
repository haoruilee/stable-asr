import json

from stable_asr.references import (
    audit_reference_workqueue_evidence,
    audit_reference_assignments,
    reference_workqueue_assignments,
    reference_workqueue_assignments_markdown,
    reference_workqueue_assignments_tsv,
    reference_workqueue_evidence_markdown,
    reference_workqueue_from_registries,
    reference_workqueue_jsonl,
    reference_workqueue_markdown,
    validate_reference_workqueue,
)


def test_reference_workqueue_merges_asr_and_turn_sources() -> None:
    workqueue = reference_workqueue_from_registries()
    validation = validate_reference_workqueue(workqueue)

    assert validation.ok
    task_ids = {task["task_id"] for task in workqueue["tasks"]}
    assert "asr:funasr" in task_ids
    assert "turn:smart_turn" in task_ids
    assert "asr:kaldi" in task_ids
    assert "asr:whisperkit" not in task_ids
    assert {task["priority"] for task in workqueue["tasks"]}.issubset({"p0", "p1"})
    assert any(task["license_review_required"] for task in workqueue["tasks"])


def test_reference_workqueue_can_filter_to_p0_only() -> None:
    workqueue = reference_workqueue_from_registries(required_priorities=("p0",))

    assert {task["priority"] for task in workqueue["tasks"]} == {"p0"}
    assert "asr:funasr" in {task["task_id"] for task in workqueue["tasks"]}
    assert "asr:kaldi" not in {task["task_id"] for task in workqueue["tasks"]}


def test_reference_workqueue_markdown_and_jsonl_render() -> None:
    workqueue = reference_workqueue_from_registries()
    markdown = reference_workqueue_markdown(workqueue)
    evidence_templates = reference_workqueue_evidence_markdown(workqueue)
    jsonl = reference_workqueue_jsonl(workqueue)

    assert "# Stable-ASR Reference Work Queue" in markdown
    assert "runs/final/asr_commands/raw/funasr_raw.jsonl" in markdown
    assert "license_review" in markdown
    assert "Stable-ASR Reference Evidence Templates" in evidence_templates
    assert "Acceptance Rule" in evidence_templates
    assert "--require-content" in evidence_templates
    assert "asr:funasr" in evidence_templates
    assert "runs/final/asr_commands/raw/funasr_raw.jsonl" in evidence_templates
    rows = [json.loads(line) for line in jsonl.splitlines()]
    assert len(rows) == len(workqueue["tasks"])
    assert rows[0]["task_id"]


def test_reference_workqueue_evidence_audit_reports_missing_targets() -> None:
    workqueue = reference_workqueue_from_registries(required_priorities=("p0",))

    report = audit_reference_workqueue_evidence(workqueue)

    assert not report.ok
    assert report.require_content is False
    assert any(item.startswith("asr:funasr:") for item in report.missing_evidence)
    assert any(item.startswith("turn:smart_turn:") for item in report.missing_evidence)
    assert any(item.startswith("asr:funasr:") for item in report.missing_license_reviews)
    assert "Reference Evidence Audit" in report.to_markdown()


def test_reference_workqueue_evidence_audit_accepts_ready_targets(tmp_path) -> None:
    workqueue = {
        "id": "stable_asr_reference_workqueue_v0",
        "version": "0.1.0",
        "generated_by": "unit",
        "required_priorities": ["p0"],
        "tasks": [
            {
                "task_id": "asr:unit",
                "collection_type": "asr",
                "reference_id": "unit",
                "name": "Unit ASR",
                "category": "unit",
                "priority": "p0",
                "acquisition_track": "ASR command adapter",
                "evidence_target": "runs/collections/unit/EVIDENCE.md",
                "license": "see_upstream",
                "license_review_required": True,
                "license_review_target": "runs/collections/unit/LICENSE_REVIEW.md",
                "policy": "link_or_command_adapter_until_reviewed",
                "status": "link_or_command_adapter_until_license_review",
                "next_action": "write_evidence",
                "blocked_by": ["license_review_before_vendoring"],
                "source_url": "https://example.com/unit",
                "docs_url": "https://example.com/unit/docs",
                "stable_asr_actions": ["write_evidence"],
                "reference_use": "unit test",
            }
        ],
    }
    (tmp_path / "runs" / "collections" / "unit").mkdir(parents=True)
    (tmp_path / "runs" / "collections" / "unit" / "EVIDENCE.md").write_text("evidence\n", encoding="utf-8")
    (tmp_path / "runs" / "collections" / "unit" / "LICENSE_REVIEW.md").write_text("review\n", encoding="utf-8")

    report = audit_reference_workqueue_evidence(workqueue, repo_root=tmp_path)

    assert report.ok
    assert not report.missing_evidence
    assert not report.missing_license_reviews


def test_reference_workqueue_evidence_audit_strict_content_rejects_empty_templates(tmp_path) -> None:
    workqueue = {
        "id": "stable_asr_reference_workqueue_v0",
        "version": "0.1.0",
        "generated_by": "unit",
        "required_priorities": ["p0"],
        "tasks": [
            {
                "task_id": "asr:unit",
                "collection_type": "asr",
                "reference_id": "unit",
                "name": "Unit ASR",
                "category": "unit",
                "priority": "p0",
                "acquisition_track": "ASR command adapter",
                "evidence_target": "runs/collections/unit/EVIDENCE.md",
                "license": "see_upstream",
                "license_review_required": True,
                "license_review_target": "runs/collections/unit/LICENSE_REVIEW.md",
                "policy": "link_or_command_adapter_until_reviewed",
                "status": "link_or_command_adapter_until_license_review",
                "next_action": "write_evidence",
                "blocked_by": ["license_review_before_vendoring"],
                "source_url": "https://example.com/unit",
                "docs_url": "https://example.com/unit/docs",
                "stable_asr_actions": ["write_evidence"],
                "reference_use": "unit test",
            }
        ],
    }
    root = tmp_path / "runs" / "collections" / "unit"
    root.mkdir(parents=True)
    (root / "EVIDENCE.md").write_text(
        "\n".join(
            [
                "# Evidence",
                "",
                "## Upstream version and source",
                "",
                "## Inputs used",
                "",
                "## Command, script, or bridge implementation notes",
                "",
                "## Output paths and schema or validation commands",
                "",
                "## Metrics, examples, or failure notes relevant to Stable-ASR",
                "",
                "## License and redistribution decision",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "LICENSE_REVIEW.md").write_text(
        "# License Review: Unit\n\n## Decision\n\n- status: pending\n- reviewer:\n- approved_uses:\n- prohibited_uses:\n- required_notices:\n",
        encoding="utf-8",
    )

    report = audit_reference_workqueue_evidence(workqueue, repo_root=tmp_path, require_content=True)

    assert not report.ok
    assert report.require_content is True
    assert report.incomplete_evidence
    assert report.incomplete_license_reviews
    assert "has no filled content" in report.incomplete_evidence[0]
    assert "decision status is still pending" in report.incomplete_license_reviews[0]


def test_reference_workqueue_evidence_audit_strict_content_accepts_jsonl_and_review(tmp_path) -> None:
    workqueue = {
        "id": "stable_asr_reference_workqueue_v0",
        "version": "0.1.0",
        "generated_by": "unit",
        "required_priorities": ["p0"],
        "tasks": [
            {
                "task_id": "asr:unit",
                "collection_type": "asr",
                "reference_id": "unit",
                "name": "Unit ASR",
                "category": "unit",
                "priority": "p0",
                "acquisition_track": "ASR command adapter",
                "evidence_target": "runs/collections/unit/raw.jsonl",
                "license": "see_upstream",
                "license_review_required": True,
                "license_review_target": "runs/collections/unit/LICENSE_REVIEW.md",
                "policy": "link_or_command_adapter_until_reviewed",
                "status": "link_or_command_adapter_until_license_review",
                "next_action": "write_evidence",
                "blocked_by": ["license_review_before_vendoring"],
                "source_url": "https://example.com/unit",
                "docs_url": "https://example.com/unit/docs",
                "stable_asr_actions": ["write_evidence"],
                "reference_use": "unit test",
            }
        ],
    }
    root = tmp_path / "runs" / "collections" / "unit"
    root.mkdir(parents=True)
    (root / "raw.jsonl").write_text('{"id":"utt1","text":"hello"}\n', encoding="utf-8")
    (root / "LICENSE_REVIEW.md").write_text(
        "# License Review: Unit\n\n## Decision\n\n- status: approved\n- reviewer: reviewer\n- approved_uses: command adapter outputs\n- prohibited_uses: vendored weights\n- required_notices: cite upstream\n",
        encoding="utf-8",
    )

    report = audit_reference_workqueue_evidence(workqueue, repo_root=tmp_path, require_content=True)

    assert report.ok
    assert not report.incomplete_evidence
    assert not report.incomplete_license_reviews
    assert report.rows[0].evidence_content_checked
    assert report.rows[0].license_review_content_checked


def test_reference_workqueue_assignment_templates_render() -> None:
    workqueue = reference_workqueue_from_registries()
    assignments = reference_workqueue_assignments(workqueue)
    markdown = reference_workqueue_assignments_markdown(assignments)
    tsv = reference_workqueue_assignments_tsv(assignments)

    assert assignments["id"] == "stable_asr_reference_assignments_v0"
    assert any(row["task_id"] == "asr:funasr" and row["blocking_release"] for row in assignments["rows"])
    assert any(row["task_id"] == "turn:smart_turn" for row in assignments["rows"])
    assert "Stable-ASR Reference Assignments" in markdown
    assert "Owner Workflow" in markdown
    assert "task_id\tcollection_type\treference_id" in tsv
    assert "blocked_license_review" in tsv


def test_reference_assignment_audit_flags_coordination_gaps(tmp_path) -> None:
    workqueue = reference_workqueue_from_registries()
    assignments = tmp_path / "reference_assignments.json"
    assignments.write_text(json.dumps(reference_workqueue_assignments(workqueue), ensure_ascii=False, indent=2) + "\n")

    report = audit_reference_assignments(assignments, repo_root=tmp_path)
    strict = audit_reference_assignments(
        assignments,
        repo_root=tmp_path,
        require_owner=True,
        require_due_date=True,
        require_ready=True,
    )

    assert report.ok
    assert "asr:funasr" in report.unassigned
    assert "asr:funasr" in report.missing_due_dates
    assert "asr:funasr" in report.blocking_release
    assert "asr:funasr" in report.missing_evidence
    assert "asr:funasr" in report.missing_license_reviews
    assert "Stable-ASR Reference Assignment Audit" in report.to_markdown()
    assert not strict.ok
    assert "asr:funasr:owner:unassigned" in strict.errors
    assert "asr:funasr:due_date:missing" in strict.errors
    assert "asr:funasr:blocking_release" in strict.errors
