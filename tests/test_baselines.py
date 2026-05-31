from stable_asr.data.manifest import load_manifest
from stable_asr.models.baselines import RuleEndpointBaseline, TextTurnBaseline, VADPauseBaseline
from stable_asr.scenarios.synthetic_turn import generate_synthetic_turn_records


def test_rule_endpoint_baseline_reads_pause_metadata() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    baseline = RuleEndpointBaseline(complete_pause_ms=700)

    assert baseline.predict(records[0]).label == "complete"
    assert baseline.predict(records[1]).label == "incomplete"


def test_vad_pause_baseline_reads_vad_pause_metadata() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    baseline = VADPauseBaseline(complete_pause_ms=700)

    assert baseline.predict(records[0]).label == "complete"
    assert baseline.predict(records[1]).label == "incomplete"


def test_text_turn_baseline_reads_transcript_cues() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    baseline = TextTurnBaseline()

    assert baseline.predict(records[0]).label == "complete"
    assert baseline.predict(records[1]).label == "incomplete"
    assert baseline.predict(records[2]).label == "backchannel"
    assert baseline.predict(records[3]).label == "wait"


def test_text_turn_baseline_treats_corrections_as_interruptions() -> None:
    records = generate_synthetic_turn_records(5, seed=0)
    interruption = [record for record in records if record.scenario == "user_interruption"][0]

    assert TextTurnBaseline().predict(interruption).label == "complete"
