from pathlib import Path

import pytest

import stable_asr as sasr
from stable_asr.data.manifest import load_manifest
from stable_asr.models.baselines import TextTurnBaseline


def test_world_collect_and_evaluate_manifest(tmp_path: Path) -> None:
    world = sasr.World("stable_asr_voiceworld_v0", num_envs=2, seed=7)
    manifest = tmp_path / "voiceworld.jsonl"

    records = world.collect(manifest, episodes=9)
    loaded = load_manifest(manifest)
    report = world.evaluate(TextTurnBaseline(), dataset=manifest)

    assert world.spec.suite_id == "stable_asr_voiceworld_v0"
    assert world.num_envs == 2
    assert len(records) == 9
    assert records[0].source == "stable_asr_voiceworld_v0"
    assert [record.id for record in loaded] == [record.id for record in records]
    assert set(world.scenarios) == {record.scenario for record in records}
    assert report.suite == "stable_asr_voiceworld_v0"
    assert len(report.overall.examples) == 9


def test_world_alias_and_default_baseline_are_seedable() -> None:
    world = sasr.World("sdx/zh-full-duplex-mini-v1", seed=11)

    first = world.sample(episodes=6)
    second = world.sample(episodes=6)
    report = world.evaluate(baseline="text_turn", episodes=6)

    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]
    assert report.suite == "stable_asr_voiceworld_v0"
    assert "normal_question" in report.by_scenario


def test_world_rejects_unknown_suite_and_baseline() -> None:
    with pytest.raises(ValueError, match="unknown VoiceWorld suite"):
        sasr.World("missing/world")

    world = sasr.World()
    with pytest.raises(ValueError, match="unknown baseline"):
        world.evaluate(baseline="missing")
