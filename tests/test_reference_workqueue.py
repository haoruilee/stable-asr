import json

from stable_asr.references import (
    audit_reference_assignments,
    reference_workqueue_assignments,
    reference_workqueue_assignments_markdown,
    reference_workqueue_assignments_tsv,
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
    jsonl = reference_workqueue_jsonl(workqueue)

    assert "# Stable-ASR Reference Work Queue" in markdown
    assert "runs/final/asr_commands/raw/funasr_raw.jsonl" in markdown
    assert "license_review" in markdown
    rows = [json.loads(line) for line in jsonl.splitlines()]
    assert len(rows) == len(workqueue["tasks"])
    assert rows[0]["task_id"]


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
