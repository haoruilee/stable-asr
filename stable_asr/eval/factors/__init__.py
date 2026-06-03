"""Factor-of-variation encoders for the v1 paper evaluation protocol.

Each factor takes an existing audio record and produces a perturbed copy
with the perturbation parameters logged in the resulting ScenarioRecord's
metadata. Perturbation parameters are *known* (physically controlled)
rather than *measured*, so they serve as ground truth for the factor
without requiring new human annotation.

Factors implemented in this module (LDC-independent):

    F3 speech_rate  — torchaudio time-stretch resampling, controlled rate
    F4 snr          — noise injection at controlled SNR (dB)
    F5 overlap      — controlled-ratio two-speaker mixing

Factors deferred until LDC corpora arrive:

    F1 language     — LDC corpus selection (zero code; just data routing)
    F2 channel      — LDC channel metadata + ``channel_simulate`` for
                      synthetic phone-band/codec degradation
    F6 code_switch  — selection from LDC CallHome zh/ja language tags

See ``stable_asr/eval/scenario_record.py`` for the schema layered on top
of TurnManifestRecord and StreamingASRRecord, and
``stable_asr/eval/factors/channel_simulate.py`` for the in-tree
synthetic-channel encoder usable today on AMI / LibriSpeech.
"""

from stable_asr.eval.factors.channel_simulate import (
    ChannelSimulateConfig,
    apply_channel_simulate,
)
from stable_asr.eval.factors.overlap import OverlapConfig, apply_overlap
from stable_asr.eval.factors.snr import SNRConfig, apply_snr
from stable_asr.eval.factors.speech_rate import SpeechRateConfig, apply_speech_rate

__all__ = [
    "ChannelSimulateConfig",
    "OverlapConfig",
    "SNRConfig",
    "SpeechRateConfig",
    "apply_channel_simulate",
    "apply_overlap",
    "apply_snr",
    "apply_speech_rate",
]
