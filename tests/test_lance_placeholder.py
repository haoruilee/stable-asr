from pathlib import Path
import importlib

import pytest

from stable_asr.data.registry import convert_turn_manifest, load_turn_records


def test_lance_backend_roundtrips_when_available(tmp_path: Path) -> None:
    if not _has_pylance():
        pytest.skip("pylance is not installed")

    dest = tmp_path / "turn_demo.lance"
    count = convert_turn_manifest("examples/data/turn_demo.jsonl", dest)
    loaded = load_turn_records(dest)

    assert count == 4
    assert dest.exists()
    assert [record.id for record in loaded] == [
        "zh_turn_000001",
        "zh_turn_000002",
        "zh_turn_000003",
        "zh_turn_000004",
    ]


def test_lance_backend_has_clear_optional_dependency_error(tmp_path: Path) -> None:
    if _has_pylance():
        pytest.skip("pylance is installed")

    with pytest.raises(RuntimeError, match="Lance support requires"):
        convert_turn_manifest("examples/data/turn_demo.jsonl", tmp_path / "turn_demo.lance")

    with pytest.raises(RuntimeError, match="Lance support requires"):
        load_turn_records(tmp_path / "turn_demo.lance")


def _has_pylance() -> bool:
    spec = importlib.util.find_spec("lance")
    if spec is None:
        return False
    import lance

    return hasattr(lance, "dataset") and hasattr(lance, "write_dataset")
