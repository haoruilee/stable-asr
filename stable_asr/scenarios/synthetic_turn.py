"""Deterministic synthetic turn-taking manifest generation."""

from __future__ import annotations

import random
from pathlib import Path

from stable_asr.data.audio import synth_tone, write_wav_mono
from stable_asr.data.formats.jsonl import write_jsonl
from stable_asr.data.manifest import TurnManifestRecord

SCENARIO_NAMES = (
    "normal_question",
    "incomplete_pause",
    "backchannel",
    "wait_stop",
    "user_interruption",
    "side_conversation",
    "ambient_speech",
    "noisy_farfield",
    "code_switching",
)

_TEXT_BY_SCENARIO = {
    "normal_question": "我想问一下今天北京的天气",
    "incomplete_pause": "我想问一下今天北京",
    "backchannel": "嗯嗯",
    "wait_stop": "先别说我想一下",
    "user_interruption": "等一下不是这个",
    "side_conversation": "你等会儿我跟他说一下",
    "ambient_speech": "电视里有人在讲话",
    "noisy_farfield": "喂你能听清楚我说话吗",
    "code_switching": "帮我 book 一下 meeting room",
}


def generate_synthetic_turn_records(
    episodes: int,
    *,
    seed: int = 0,
    language: str = "zh",
    source: str = "synthetic_turn_v0",
    audio_format: str = "flac",
) -> list[TurnManifestRecord]:
    """Generate a deterministic manifest with scenario and factor metadata."""

    if episodes < 0:
        raise ValueError("episodes must be non-negative")

    rng = random.Random(seed)
    records: list[TurnManifestRecord] = []
    for index in range(episodes):
        scenario = SCENARIO_NAMES[index % len(SCENARIO_NAMES)]
        record_id = f"{source}_{index:06d}"
        factors = _sample_factors(rng, scenario)
        audio_ext = "wav" if audio_format == "wav" else "flac"
        record_language = "zh_en" if scenario == "code_switching" else language
        records.append(
            TurnManifestRecord.from_dict(
                {
                    "id": record_id,
                    "audio": f"audio/{record_id}.{audio_ext}",
                    "sample_rate": 16000,
                    "start": 0.0,
                    "end": factors["duration_ms"] / 1000.0,
                    "text": _TEXT_BY_SCENARIO[scenario],
                    "asr_text": _TEXT_BY_SCENARIO[scenario],
                    "turn_label": _turn_label(scenario),
                    "action_label": _action_label(scenario),
                    "assistant_speaking": scenario in {"backchannel", "user_interruption"},
                    "overlap": scenario in {
                        "backchannel",
                        "user_interruption",
                        "side_conversation",
                        "ambient_speech",
                    },
                    "scenario": scenario,
                    "language": record_language,
                    "source": source,
                    "metadata": factors,
                }
            )
        )
    return records


def write_synthetic_turn_manifest(
    path: str | Path,
    episodes: int,
    *,
    seed: int = 0,
    language: str = "zh",
    source: str = "synthetic_turn_v0",
    write_audio: bool = False,
) -> list[TurnManifestRecord]:
    records = generate_synthetic_turn_records(
        episodes,
        seed=seed,
        language=language,
        source=source,
        audio_format="wav" if write_audio else "flac",
    )
    if write_audio:
        base_dir = Path(path).parent
        for index, record in enumerate(records):
            _write_record_audio(base_dir / record.audio, record, seed=seed + index)
    write_jsonl(path, [record.to_dict() for record in records])
    return records


def _turn_label(scenario: str) -> str:
    return {
        "normal_question": "complete",
        "incomplete_pause": "incomplete",
        "backchannel": "backchannel",
        "wait_stop": "wait",
        "user_interruption": "complete",
        "side_conversation": "wait",
        "ambient_speech": "wait",
        "noisy_farfield": "complete",
        "code_switching": "complete",
    }[scenario]


def _action_label(scenario: str) -> str:
    return {
        "normal_question": "take_turn",
        "incomplete_pause": "keep_listening",
        "backchannel": "continue_speaking",
        "wait_stop": "hold",
        "user_interruption": "stop_tts_and_listen",
        "side_conversation": "ignore",
        "ambient_speech": "ignore",
        "noisy_farfield": "take_turn",
        "code_switching": "take_turn",
    }[scenario]


def _sample_factors(rng: random.Random, scenario: str) -> dict[str, object]:
    pause_ms = {
        "normal_question": rng.randint(760, 1200),
        "incomplete_pause": rng.randint(180, 520),
        "backchannel": rng.randint(60, 220),
        "wait_stop": rng.randint(900, 1800),
        "user_interruption": rng.randint(0, 180),
        "side_conversation": rng.randint(0, 260),
        "ambient_speech": rng.randint(0, 420),
        "noisy_farfield": rng.randint(650, 1200),
        "code_switching": rng.randint(700, 1300),
    }[scenario]
    duration_ms = {
        "normal_question": rng.randint(1600, 2600),
        "incomplete_pause": rng.randint(1200, 2200),
        "backchannel": rng.randint(400, 900),
        "wait_stop": rng.randint(1200, 2200),
        "user_interruption": rng.randint(800, 1500),
        "side_conversation": rng.randint(1400, 2800),
        "ambient_speech": rng.randint(1600, 3200),
        "noisy_farfield": rng.randint(1800, 3200),
        "code_switching": rng.randint(1400, 2600),
    }[scenario]
    snr_db = rng.choice([-12, -8, -5, 0]) if scenario == "noisy_farfield" else rng.choice([-5, 0, 5, 10, 20])
    reverb = rng.choice(["large_room", "hallway"]) if scenario == "noisy_farfield" else rng.choice(["none", "small_room", "large_room"])
    return {
        "pause_ms": pause_ms,
        "vad_pause_ms": max(0, pause_ms + rng.randint(-40, 40)),
        "duration_ms": duration_ms,
        "snr_db": snr_db,
        "reverb": reverb,
        "speaking_rate": rng.choice([0.8, 1.0, 1.2, 1.5]),
        "overlap_offset_ms": rng.choice([0, 100, 300, 500]),
        "network_jitter_ms": rng.choice([0, 50, 100, 300]),
        "farfield_distance_m": rng.choice([2.0, 3.5, 5.0]) if scenario == "noisy_farfield" else 0.5,
        "code_switch_ratio": rng.choice([0.25, 0.4, 0.55]) if scenario == "code_switching" else 0.0,
        "accent": rng.choice(["standard", "regional", "non_native"]),
    }


def _write_record_audio(path: Path, record: TurnManifestRecord, *, seed: int) -> None:
    frequency = {
        "normal_question": 220.0,
        "incomplete_pause": 260.0,
        "backchannel": 420.0,
        "wait_stop": 180.0,
        "user_interruption": 330.0,
        "side_conversation": 510.0,
        "ambient_speech": 130.0,
        "noisy_farfield": 150.0,
        "code_switching": 300.0,
    }.get(record.scenario or "", 220.0)
    snr_db = float(record.metadata.get("snr_db", 20.0))
    noise = 0.04 if snr_db >= 10 else 0.12 if snr_db >= 0 else 0.20
    samples = synth_tone(
        record.duration,
        sample_rate=record.sample_rate,
        frequency=frequency,
        amplitude=0.25,
        noise=noise,
        seed=seed,
    )
    write_wav_mono(path, samples, sample_rate=record.sample_rate)
