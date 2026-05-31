from pathlib import Path

from stable_asr.paper.adapter_pack import build_adapter_pack
from stable_asr.streaming.command_compare import audit_asr_command_config, compare_asr_commands_from_config


def test_build_adapter_pack_writes_external_asr_starter_files(tmp_path: Path) -> None:
    report = build_adapter_pack(tmp_path / "adapter_pack")

    assert report.ok
    assert report.adapter_registry_ok
    assert report.asr_collections_ok
    assert report.schema_registry_ok
    assert report.asr_manifest_ok
    assert report.streaming_fixture_ok
    assert report.command_config_ok
    assert report.reference_coverage_ok

    output_dir = Path(report.output_dir)
    assert (output_dir / "README.md").exists()
    assert (output_dir / "COMMANDS.md").exists()
    assert (output_dir / "commands.sh").exists()
    assert (output_dir / "configs" / "adapter_registry.json").exists()
    assert (output_dir / "configs" / "asr_collections.json").exists()
    assert (output_dir / "configs" / "asr_command_compare.json").exists()
    assert (output_dir / "data" / "asr_eval_manifest.jsonl").exists()
    assert (output_dir / "scripts" / "export_streaming_template.py").exists()
    assert "stable-asr compare-asr-commands" in (output_dir / "COMMANDS.md").read_text(encoding="utf-8")


def test_adapter_pack_command_config_audits_and_runs(tmp_path: Path) -> None:
    report = build_adapter_pack(tmp_path / "adapter_pack")
    output_dir = Path(report.output_dir)
    config = output_dir / "configs" / "asr_command_compare.json"

    audit = audit_asr_command_config(
        config,
        repo_root=output_dir,
        min_adapters=2,
        require_input_manifest=True,
    )
    comparison = compare_asr_commands_from_config(config)

    assert audit.ok
    assert comparison.rows[0].adapter == "balanced_template"
    assert comparison.rows[1].adapter == "fast_unstable_template"
    assert comparison.rows[1].report.wer >= comparison.rows[0].report.wer
