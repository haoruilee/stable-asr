"""Data loading and validation helpers."""

from stable_asr.data.asr_manifest import (
    ASRManifestError,
    ASRManifestRecord,
    ASRManifestValidationReport,
    load_asr_manifest,
    summarize_asr_records,
    validate_asr_manifest,
    write_asr_manifest,
)
from stable_asr.data.audio import load_wav_mono, synth_tone, write_wav_mono
from stable_asr.data.audio_window_cache import (
    AudioWindowBenchmarkRow,
    benchmark_audio_window_formats,
    materialize_audio_windows,
)
from stable_asr.data.benchmark import DataBenchmarkRow, benchmark_data_formats
from stable_asr.data.converters import EXTERNAL_SCHEMAS, convert_external_jsonl, convert_rows
from stable_asr.data.manifest import (
    ManifestError,
    ManifestValidationReport,
    TurnManifestRecord,
    load_manifest,
    validate_manifest,
)
from stable_asr.data.recipes import PUBLIC_ASR_CORPORA, prepare_asr_manifest, prepare_public_asr_manifest
from stable_asr.data.registry import (
    TURN_FORMATS,
    convert_turn_manifest,
    load_turn_records,
    summarize_records,
    write_turn_records,
)
from stable_asr.data.sources import (
    DEFAULT_DATA_SOURCES,
    DataSourceRegistryValidation,
    data_sources_markdown,
    load_data_sources,
    validate_data_sources,
    write_data_sources_json,
)

__all__ = [
    "ASRManifestError",
    "ASRManifestRecord",
    "ASRManifestValidationReport",
    "AudioWindowBenchmarkRow",
    "ManifestError",
    "ManifestValidationReport",
    "TURN_FORMATS",
    "DataBenchmarkRow",
    "DataSourceRegistryValidation",
    "EXTERNAL_SCHEMAS",
    "DEFAULT_DATA_SOURCES",
    "TurnManifestRecord",
    "PUBLIC_ASR_CORPORA",
    "benchmark_data_formats",
    "benchmark_audio_window_formats",
    "convert_external_jsonl",
    "convert_rows",
    "convert_turn_manifest",
    "load_asr_manifest",
    "load_wav_mono",
    "load_manifest",
    "load_turn_records",
    "load_data_sources",
    "prepare_asr_manifest",
    "prepare_public_asr_manifest",
    "materialize_audio_windows",
    "summarize_asr_records",
    "summarize_records",
    "synth_tone",
    "validate_asr_manifest",
    "validate_manifest",
    "validate_data_sources",
    "data_sources_markdown",
    "write_asr_manifest",
    "write_data_sources_json",
    "write_wav_mono",
    "write_turn_records",
]
