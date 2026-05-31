"""Stable-worldmodel-style VoiceWorld API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord, load_manifest
from stable_asr.eval.turn_eval import TurnPredictor
from stable_asr.models.baselines import RuleEndpointBaseline, TextTurnBaseline, VADPauseBaseline
from stable_asr.scenarios.suites import load_scenario_suite, validate_scenario_suite
from stable_asr.scenarios.synthetic_turn import generate_synthetic_turn_records, write_synthetic_turn_manifest
from stable_asr.scenarios.voice_world import ScenarioEvalReport, evaluate_voice_world_records
from stable_asr.turn.policy import TurnPolicyConfig


_DEFAULT_WORLD_ALIASES = {
    "stable_asr_voiceworld_v0",
    "voiceworld/v0",
    "sasr/voiceworld-v0",
    "sdx/zh-full-duplex-mini-v1",
    "zh_turn_mini_v0",
}


@dataclass(frozen=True)
class WorldSpec:
    """Serializable description of a VoiceWorld environment."""

    name: str
    suite_id: str
    version: str
    title: str
    num_envs: int
    scenarios: list[str]
    factors: list[str]
    metrics: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "suite_id": self.suite_id,
            "version": self.version,
            "title": self.title,
            "num_envs": self.num_envs,
            "scenarios": self.scenarios,
            "factors": self.factors,
            "metrics": self.metrics,
        }


class World:
    """Seedable full-duplex turn-taking scenario world.

    This is the high-level Python API intended to mirror the ergonomic shape of
    stable-worldmodel's `World(...)` entrypoint while staying specific to
    Stable-ASR's VoiceWorld scenarios.
    """

    def __init__(
        self,
        name: str = "stable_asr_voiceworld_v0",
        *,
        num_envs: int = 1,
        seed: int = 0,
        suite_path: str | Path | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.name = name
        self.num_envs = num_envs
        self.seed = seed
        self.suite = _load_suite(name, suite_path=suite_path)
        validation = validate_scenario_suite(self.suite)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))

    @property
    def spec(self) -> WorldSpec:
        return WorldSpec(
            name=self.name,
            suite_id=str(self.suite["id"]),
            version=str(self.suite["version"]),
            title=str(self.suite["title"]),
            num_envs=self.num_envs,
            scenarios=[str(item["id"]) for item in self.suite.get("scenarios", [])],
            factors=[str(item["name"]) for item in self.suite.get("factors", [])],
            metrics=[str(item) for item in self.suite.get("metrics", [])],
        )

    @property
    def scenarios(self) -> list[str]:
        return self.spec.scenarios

    @property
    def factors(self) -> list[str]:
        return self.spec.factors

    @property
    def metrics(self) -> list[str]:
        return self.spec.metrics

    def to_dict(self) -> dict[str, object]:
        return self.spec.to_dict()

    def sample(
        self,
        *,
        episodes: int = 25,
        seed: int | None = None,
        language: str = "zh",
    ) -> list[TurnManifestRecord]:
        """Sample deterministic synthetic VoiceWorld turn records."""

        return generate_synthetic_turn_records(
            episodes,
            seed=self.seed if seed is None else seed,
            language=language,
            source=str(self.suite["id"]),
        )

    def collect(
        self,
        output: str | Path,
        *,
        episodes: int = 100,
        seed: int | None = None,
        language: str = "zh",
        write_audio: bool = False,
    ) -> list[TurnManifestRecord]:
        """Write a turn manifest for this world and return the records."""

        return write_synthetic_turn_manifest(
            output,
            episodes,
            seed=self.seed if seed is None else seed,
            language=language,
            source=str(self.suite["id"]),
            write_audio=write_audio,
        )

    def evaluate(
        self,
        predictor: TurnPredictor | None = None,
        *,
        baseline: str = "vad_pause",
        dataset: str | Path | None = None,
        episodes: int = 25,
        seed: int | None = None,
        policy_config: TurnPolicyConfig | None = None,
    ) -> ScenarioEvalReport:
        """Evaluate a predictor on synthetic or manifest-backed VoiceWorld records."""

        records = load_manifest(dataset) if dataset is not None else self.sample(episodes=episodes, seed=seed)
        return evaluate_voice_world_records(
            records,
            predictor or _baseline_predictor(baseline),
            seed=self.seed if seed is None else seed,
            suite=str(self.suite["id"]),
            policy_config=policy_config,
        )


VoiceWorld = World


def _load_suite(name: str, *, suite_path: str | Path | None) -> dict[str, Any]:
    if suite_path is not None:
        return load_scenario_suite(suite_path)
    candidate = Path(name)
    if candidate.exists():
        return load_scenario_suite(candidate)
    if name in _DEFAULT_WORLD_ALIASES:
        return load_scenario_suite()
    raise ValueError(
        f"unknown VoiceWorld suite {name!r}; pass suite_path=... for a custom scenario suite"
    )


def _baseline_predictor(name: str) -> TurnPredictor:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"vad", "vad_pause", "vad_pause_baseline"}:
        return VADPauseBaseline()
    if normalized in {"rule", "rule_endpoint", "rule_endpoint_baseline"}:
        return RuleEndpointBaseline()
    if normalized in {"text", "text_turn", "text_turn_baseline"}:
        return TextTurnBaseline()
    raise ValueError(f"unknown baseline {name!r}")
