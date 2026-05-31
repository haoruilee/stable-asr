from pathlib import Path

from stable_asr.references import load_asr_collections
from stable_asr.data.registry import load_turn_records
from stable_asr.resources import resolve_platform_path
from stable_asr.roadmap import load_roadmap


def test_resolve_platform_path_prefers_existing_repo_path() -> None:
    path = resolve_platform_path("configs/roadmap/stable_asr_roadmap.json")

    assert path == Path("configs/roadmap/stable_asr_roadmap.json")


def test_resolve_platform_path_falls_back_to_asset_root(tmp_path, monkeypatch) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    asset_root = tmp_path / "assets"
    asset = asset_root / "configs" / "references"
    asset.mkdir(parents=True)
    target = asset / "asr_collections.json"
    target.write_text('{"id":"x","version":"0","reviewed_at":"today","title":"X","entries":[]}\n', encoding="utf-8")
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("STABLE_ASR_ASSET_ROOT", str(asset_root))

    resolved = resolve_platform_path("configs/references/asr_collections.json")

    assert resolved == target


def test_resolve_platform_path_falls_back_to_source_root(tmp_path, monkeypatch) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    resolved = resolve_platform_path("configs/roadmap/stable_asr_roadmap.json")

    assert resolved == Path(__file__).resolve().parents[1] / "configs" / "roadmap" / "stable_asr_roadmap.json"


def test_default_config_loaders_accept_explicit_repo_paths() -> None:
    assert load_roadmap("configs/roadmap/stable_asr_roadmap.json")["id"] == "stable_asr_roadmap_v0"
    assert load_asr_collections("configs/references/asr_collections.json")["id"] == "stable_asr_reference_collections_v0"


def test_turn_record_loader_resolves_platform_paths(tmp_path, monkeypatch) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    records = load_turn_records("examples/data/turn_demo.jsonl")

    assert len(records) == 4
