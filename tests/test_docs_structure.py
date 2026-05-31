from pathlib import Path


def test_mkdocs_nav_targets_exist() -> None:
    mkdocs = Path("mkdocs.yaml").read_text(encoding="utf-8")
    expected = [
        "docs/index.md",
        "docs/quick_start.md",
        "docs/cli.md",
        "docs/baselines.md",
        "docs/schema.md",
        "docs/paper_pipeline.md",
        "docs/release_gates.md",
        "docs/final_inputs.md",
        "docs/asr_collections.md",
        "docs/api/data.md",
        "docs/api/turn.md",
        "docs/api/scenarios.md",
        "docs/api/paper.md",
        "docs/guides/adapters.md",
        "docs/guides/release_smoke.md",
    ]

    for path in expected:
        nav_target = path.removeprefix("docs/")
        assert nav_target in mkdocs
        assert Path(path).exists()


def test_docs_index_links_platform_sections() -> None:
    text = Path("docs/index.md").read_text(encoding="utf-8")

    assert "quick_start.md" in text
    assert "cli.md" in text
    assert "baselines.md" in text
    assert "final_inputs.md" in text
    assert "api/data.md" in text
    assert "guides/release_smoke.md" in text
