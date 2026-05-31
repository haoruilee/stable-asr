from dataclasses import replace

from stable_asr.data.registry import load_turn_records
from stable_asr.data.split import TurnSplitConfig, split_turn_records


def test_split_turn_records_is_deterministic() -> None:
    records = load_turn_records("examples/data/turn_demo.jsonl")

    first = split_turn_records(records, config=TurnSplitConfig(train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=7))
    second = split_turn_records(records, config=TurnSplitConfig(train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=7))

    assert [record.id for record in first.train] == [record.id for record in second.train]
    assert [record.id for record in first.dev] == [record.id for record in second.dev]
    assert [record.id for record in first.test] == [record.id for record in second.test]
    assert len(first.train) == 2
    assert len(first.dev) == 1
    assert len(first.test) == 1


def test_split_turn_records_keeps_groups_together() -> None:
    base = load_turn_records("examples/data/turn_demo.jsonl")
    records = []
    for group_index in range(4):
        group_id = f"group_{group_index}"
        for item_index, record in enumerate(base[:2]):
            records.append(
                replace(
                    record,
                    id=f"{group_id}_{item_index}",
                    metadata={**record.metadata, "conversation_id": group_id},
                )
            )

    result = split_turn_records(
        records,
        config=TurnSplitConfig(
            train_ratio=0.5,
            dev_ratio=0.25,
            test_ratio=0.25,
            seed=3,
            stratify_by=("turn_label",),
            group_by="metadata.conversation_id",
        ),
    )

    seen: dict[str, str] = {}
    for split_name in ("train", "dev", "test"):
        for record in result.split(split_name):
            group_id = str(record.metadata["conversation_id"])
            seen.setdefault(group_id, split_name)
            assert seen[group_id] == split_name
