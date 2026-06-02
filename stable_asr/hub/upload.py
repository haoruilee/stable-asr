"""HuggingFace Hub upload utilities for stable-asr datasets and models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_hf_hub():
    try:
        from huggingface_hub import HfApi
        return HfApi
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace upload requires huggingface_hub. "
            "Install with: pip install 'stable-asr[hub]'"
        ) from exc


def upload_dataset(
    manifest_path: str | Path,
    repo_id: str,
    *,
    split: str = "train",
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a JSONL or Parquet turn manifest to a HuggingFace dataset repo.

    Creates the repo if it does not exist. The manifest is uploaded as
    ``data/{split}.jsonl`` (or ``.parquet``).

    Returns the repo URL.
    """
    HfApi = _require_hf_hub()
    from huggingface_hub import HfApi as _HfApi

    api = _HfApi(token=token)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    suffix = manifest_path.suffix.lower()
    dest = f"data/{split}{suffix}"
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo=dest,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or f"stable-asr: upload {split} split ({manifest_path.name})",
    )

    readme = _dataset_readme(repo_id, split=split, manifest_name=manifest_path.name)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="stable-asr: update dataset card",
    )

    return f"https://huggingface.co/datasets/{repo_id}"


def upload_model(
    checkpoint_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    onnx_path: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Upload a NanoTurn checkpoint (.pt) to a HuggingFace model repo.

    Optionally also uploads an ONNX export and writes a model card.

    Returns the repo URL.
    """
    HfApi = _require_hf_hub()
    from huggingface_hub import HfApi as _HfApi

    api = _HfApi(token=token)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or f"stable-asr: upload {checkpoint_path.name}",
    )

    if onnx_path is not None:
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        api.upload_file(
            path_or_fileobj=str(onnx_path),
            path_in_repo=onnx_path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"stable-asr: upload ONNX export {onnx_path.name}",
        )

    readme = _model_readme(repo_id, checkpoint_name=checkpoint_path.name, metrics=metrics)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="stable-asr: update model card",
    )

    return f"https://huggingface.co/models/{repo_id}"


def upload_experiment_dir(
    experiment_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    patterns: list[str] | None = None,
) -> str:
    """Upload an entire experiment output directory to a HuggingFace model repo.

    By default uploads ``*.pt``, ``*.onnx``, ``*.json``, ``*.jsonl``,
    ``*.md``, and ``*.txt`` files. Pass ``patterns`` to override.

    Returns the repo URL.
    """
    HfApi = _require_hf_hub()
    from huggingface_hub import HfApi as _HfApi

    api = _HfApi(token=token)
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"experiment_dir not found: {experiment_dir}")

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    include_patterns = patterns or ["*.pt", "*.onnx", "*.json", "*.jsonl", "*.md", "*.txt"]
    api.upload_folder(
        folder_path=str(experiment_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=include_patterns,
        commit_message=commit_message or f"stable-asr: upload experiment dir {experiment_dir.name}",
    )

    return f"https://huggingface.co/models/{repo_id}"


# ---------------------------------------------------------------------------
# Card templates
# ---------------------------------------------------------------------------


def _dataset_readme(repo_id: str, *, split: str, manifest_name: str) -> str:
    return f"""\
---
license: mit
task_categories:
  - audio-classification
language:
  - en
tags:
  - speech
  - turn-taking
  - endpointing
  - stable-asr
---

# {repo_id.split("/")[-1]}

Turn-taking manifest dataset produced by [stable-asr](https://github.com/haoruilee/stable-asr).

## Split

| split | file |
| --- | --- |
| {split} | `data/{manifest_name}` |

## Schema

Each record is a `TurnManifestRecord` with fields: `id`, `audio`, `start`, `end`,
`sample_rate`, `turn_label`, `assistant_speaking`, `overlap`, `metadata`.

## Labels

`complete`, `incomplete`, `backchannel`, `wait`

## Citation

If you use this dataset, please cite the stable-asr platform paper.
"""


def _model_readme(
    repo_id: str,
    *,
    checkpoint_name: str,
    metrics: dict[str, Any] | None,
) -> str:
    metrics_section = ""
    if metrics:
        rows = "\n".join(
            f"| `{k}` | `{v}` |"
            for k, v in metrics.items()
            if isinstance(v, (int, float, str))
        )
        metrics_section = f"\n## Training metrics\n\n| key | value |\n| --- | --- |\n{rows}\n"

    return f"""\
---
license: mit
library_name: stable-asr
tags:
  - speech
  - turn-taking
  - endpointing
  - pytorch
---

# {repo_id.split("/")[-1]}

NanoTurn model produced by [stable-asr](https://github.com/haoruilee/stable-asr).

## Files

- `{checkpoint_name}` — PyTorch checkpoint (load with `stable_asr.train.turn_trainer.load_nanoturn_checkpoint`)
{metrics_section}
## Usage

```python
from stable_asr.train.turn_trainer import NanoTurnCheckpointPredictor
predictor = NanoTurnCheckpointPredictor("{checkpoint_name}")
prediction = predictor.predict(record)
```

## Citation

If you use this model, please cite the stable-asr platform paper.
"""
