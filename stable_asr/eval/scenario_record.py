"""ScenarioRecord — factor-aware extension of TurnManifestRecord.

A ScenarioRecord wraps an existing TurnManifestRecord and pins down which
factor-of-variation perturbations have been applied to produce the audio
referenced in ``audio``. The original (unperturbed) record's id is kept in
``base_id`` so factor-effect tables can join back to a clean baseline.

Schema rationale (read this before adding a new factor):

* ``factor`` is the high-level axis name (one of the strings in
  ``KNOWN_FACTORS``). One ScenarioRecord encodes exactly one factor's
  perturbation; if you need to compose two factors (e.g. SNR × speech-rate),
  apply them in order, each producing a new ScenarioRecord whose
  ``base_id`` points to the previous step.
* ``factor_level`` is the discrete bucket label that identifies *which*
  level along the factor axis this record represents (e.g. ``"snr_5db"``
  or ``"rate_1.3x"``). This is what later analyses group by.
* ``factor_params`` holds the exact perturbation parameters as a JSON
  object. These must be enough to reproduce the perturbation byte-for-byte
  given the source audio (they are the "ground truth" for the factor).
* ``base_id`` is the id of the unperturbed source record. Critical for
  joins.
* ``metadata`` is forwarded from TurnManifestRecord and may include the
  source manifest's own metadata.

The schema is intentionally structural — it does not restrict which
factor names are valid beyond a soft check, so new factors can be added
without breaking older fixtures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord

KNOWN_FACTORS: frozenset[str] = frozenset(
    {
        "language",         # F1 — corpus selection
        "channel",          # F2 — corpus channel metadata or simulate
        "channel_simulate", # F2' — synthetic phone-band/codec
        "speech_rate",      # F3 — controlled time-stretch
        "snr",              # F4 — controlled noise injection
        "overlap",          # F5 — controlled two-speaker mixing
        "code_switch",      # F6 — LDC language-tag selection
    }
)


@dataclass(frozen=True)
class ScenarioRecord:
    """A TurnManifestRecord plus the factor perturbation that produced it.

    ScenarioRecord serialises as a flat JSON object, so existing tooling
    that reads JSONL by ``TurnManifestRecord.from_dict`` still works on
    the same files when factor fields are absent. Use
    ``ScenarioRecord.from_dict`` to deserialize records that include
    factor metadata.
    """

    # All TurnManifestRecord fields (kept in sync; replicated rather than
    # embedded so the serialised JSON stays flat and existing readers
    # keep working).
    id: str
    audio: str
    sample_rate: int
    start: float
    end: float
    turn_label: str
    action_label: str
    assistant_speaking: bool
    overlap: bool
    language: str
    source: str
    text: str | None = None
    asr_text: str | None = None
    scenario: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Factor-of-variation extension fields.
    base_id: str | None = None
    factor: str | None = None
    factor_level: str | None = None
    factor_params: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def from_record(
        cls,
        base: TurnManifestRecord,
        *,
        audio: str,
        factor: str,
        factor_level: str,
        factor_params: dict[str, Any],
        new_id: str | None = None,
        sample_rate: int | None = None,
        start: float | None = None,
        end: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> "ScenarioRecord":
        """Build a ScenarioRecord by applying a factor to a TurnManifestRecord.

        ``audio`` is the path to the perturbed audio; the rest of the
        manifest fields are copied through. If a perturbation changes the
        time base (e.g. speech-rate resample) the caller should pass the
        new ``sample_rate`` / ``start`` / ``end``.
        """

        if factor not in KNOWN_FACTORS:
            # Soft check — new factors can be added by extending KNOWN_FACTORS.
            raise ValueError(
                f"unknown factor {factor!r}; add it to KNOWN_FACTORS in "
                f"stable_asr.eval.scenario_record"
            )

        merged_metadata = dict(base.metadata)
        if extra_metadata:
            merged_metadata.update(extra_metadata)

        return cls(
            id=new_id or f"{base.id}__{factor}_{factor_level}",
            audio=audio,
            sample_rate=sample_rate if sample_rate is not None else base.sample_rate,
            start=start if start is not None else base.start,
            end=end if end is not None else base.end,
            turn_label=base.turn_label,
            action_label=base.action_label,
            assistant_speaking=base.assistant_speaking,
            overlap=base.overlap,
            language=base.language,
            source=base.source,
            text=base.text,
            asr_text=base.asr_text,
            scenario=base.scenario,
            metadata=merged_metadata,
            base_id=base.id,
            factor=factor,
            factor_level=factor_level,
            factor_params=dict(factor_params),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioRecord":
        return cls(
            id=str(data["id"]),
            audio=str(data["audio"]),
            sample_rate=int(data["sample_rate"]),
            start=float(data["start"]),
            end=float(data["end"]),
            turn_label=str(data["turn_label"]),
            action_label=str(data["action_label"]),
            assistant_speaking=bool(data["assistant_speaking"]),
            overlap=bool(data["overlap"]),
            language=str(data["language"]),
            source=str(data["source"]),
            text=data.get("text"),
            asr_text=data.get("asr_text"),
            scenario=data.get("scenario"),
            metadata=dict(data.get("metadata") or {}),
            base_id=data.get("base_id"),
            factor=data.get("factor"),
            factor_level=data.get("factor_level"),
            factor_params=dict(data.get("factor_params") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_turn_record(self) -> TurnManifestRecord:
        """Project to a plain TurnManifestRecord (factor fields dropped).

        Useful for feeding existing turn-taking baselines that only know
        about TurnManifestRecord.
        """

        return TurnManifestRecord(
            id=self.id,
            audio=self.audio,
            sample_rate=self.sample_rate,
            start=self.start,
            end=self.end,
            turn_label=self.turn_label,
            action_label=self.action_label,
            assistant_speaking=self.assistant_speaking,
            overlap=self.overlap,
            language=self.language,
            source=self.source,
            text=self.text,
            asr_text=self.asr_text,
            scenario=self.scenario,
            metadata=dict(self.metadata),
        )


def write_scenario_jsonl(path: str | Path, records: list[ScenarioRecord]) -> None:
    """Write ScenarioRecords as one JSON object per line."""

    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def read_scenario_jsonl(path: str | Path) -> list[ScenarioRecord]:
    """Read ScenarioRecords from JSONL (factor fields default to None)."""

    import json

    out: list[ScenarioRecord] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            out.append(ScenarioRecord.from_dict(json.loads(line)))
    return out
