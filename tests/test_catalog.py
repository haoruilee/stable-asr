import json

from stable_asr.catalog import build_platform_catalog, write_platform_catalog_json, write_platform_catalog_markdown


def test_platform_catalog_summarizes_registered_assets(tmp_path) -> None:
    report = build_platform_catalog(repo_root=".")

    assert report.ok
    payload = report.to_dict()
    assert payload["stable_worldmodel_parity"]["ok"] is True
    section_names = {section["name"] for section in payload["sections"]}
    assert "data_sources" in section_names
    assert "adapters" in section_names
    assert "voiceworld_scenarios" in section_names
    assert "asr_references" in section_names
    assert "roadmap" in section_names

    markdown = report.to_markdown()
    assert "Stable-ASR Platform Catalog" in markdown
    assert "Stable-WorldModel-Style Parity" in markdown
    assert "stable-asr final-config --config configs/final/paper_final.json --plan-missing" in markdown

    json_path = tmp_path / "catalog.json"
    markdown_path = tmp_path / "CATALOG.md"
    write_platform_catalog_json(report, json_path)
    write_platform_catalog_markdown(report, markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["ok"] is True
    assert "Registered Assets" in markdown_path.read_text(encoding="utf-8")
