"""Model adapters, baselines, and registries."""

from stable_asr.models.registry import (
    DEFAULT_MODEL_REGISTRY,
    DEFAULT_MODEL_REGISTRY_PATH,
    ModelRegistryValidation,
    find_model_entry,
    load_model_registry,
    model_registry_markdown,
    validate_model_registry,
    write_model_registry_json,
)

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "DEFAULT_MODEL_REGISTRY_PATH",
    "ModelRegistryValidation",
    "find_model_entry",
    "load_model_registry",
    "model_registry_markdown",
    "validate_model_registry",
    "write_model_registry_json",
]
