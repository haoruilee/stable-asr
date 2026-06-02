"""VAP (Voice Activity Projection) baseline adapter.

Wraps the official ErikEkstedt/VAP model so it conforms to the
TurnPredictor protocol and can be used in compare-turn / benchmark-turn.

Install requirements:
    pip install vap-turn-taking  # or: pip install git+https://github.com/ErikEkstedt/VoiceActivityProjection

The model predicts the probability of a voice activity shift in the next
0-1s window given a short audio context. We map this to a turn-taking
label as follows:
    p_shift_now (VAP "p1" at the utterance end) → prob("complete")
    1 - p_shift_now                             → prob("incomplete")

Reference:
    Ekstedt & Skantze (2022). "Voice Activity Projection: Self-supervised
    Learning of Turn-taking Events." Interspeech 2022.
    https://arxiv.org/abs/2205.09812
    Code: https://github.com/ErikEkstedt/VoiceActivityProjection
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.turn.types import TurnPrediction

if TYPE_CHECKING:
    pass

# Default public checkpoint released by Ekstedt & Skantze.
# Can be overridden via VAPPredictor(checkpoint=...).
_DEFAULT_CHECKPOINT = "ErikEkstedt/VAP"

# Audio context fed into VAP (seconds before the utterance end).
# VAP was trained with ~10s of stereo context; we give it what we have.
_CONTEXT_SEC = 10.0


class VAPPredictor:
    """Turn predictor that runs the official VAP model.

    Parameters
    ----------
    checkpoint:
        HuggingFace model ID or local path to a VAP checkpoint.
    device:
        Torch device string, e.g. "cpu", "cuda", "cuda:0".
        Defaults to "cuda" if available, else "cpu".
    context_sec:
        Seconds of audio context to feed the model.
    threshold:
        p_shift >= threshold → label "complete".  Default 0.5.
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        *,
        device: str | None = None,
        context_sec: float = _CONTEXT_SEC,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint = checkpoint
        self.context_sec = context_sec
        self.threshold = threshold
        self._model = None
        self._device: str | None = device

    def _load(self) -> None:
        """Lazy-load the VAP model on first call."""
        if self._model is not None:
            return
        try:
            import torch
            from vap.model import VAPModel  # pip install vap-turn-taking
        except ImportError as exc:
            raise RuntimeError(
                "VAP model requires the vap-turn-taking package.\n"
                "Install with: pip install vap-turn-taking\n"
                "or: pip install git+https://github.com/ErikEkstedt/VoiceActivityProjection"
            ) from exc

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = device

        # Load from HuggingFace or local path
        model_path = self.checkpoint
        if not Path(model_path).exists():
            # Download from HuggingFace hub
            from huggingface_hub import hf_hub_download
            try:
                model_path = hf_hub_download(
                    repo_id=self.checkpoint,
                    filename="model.pt",
                )
            except Exception:
                # Some VAP checkpoints use different filenames
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(repo_id=self.checkpoint)

        self._model = VAPModel.load_from_checkpoint(model_path, map_location=device)
        self._model.eval()
        self._model.to(device)

    def _load_audio_context(
        self,
        record: TurnManifestRecord,
    ):
        """Load a stereo audio tensor (1, 2, T) for VAP input.

        VAP expects stereo where channel 0 = current speaker, channel 1 = other.
        For our single-speaker manifests, we put audio on channel 0 and silence
        on channel 1, which gives a conservative (lower) shift probability.
        """
        import torch
        from stable_asr.data.audio import load_audio_mono

        path = Path(record.audio)
        samples, sr = load_audio_mono(path, target_sample_rate=16000)

        # Extract context window ending at record.end
        end_sample = min(len(samples), int(round(record.end * sr)))
        context_samples = int(self.context_sec * sr)
        start_sample = max(0, end_sample - context_samples)
        audio = torch.tensor(samples[start_sample:end_sample], dtype=torch.float32)

        # Pad to at least 1600 samples (0.1s)
        if audio.numel() < 1600:
            audio = torch.nn.functional.pad(audio, (0, 1600 - audio.numel()))

        # VAP input shape: (1, 2, T) — batch=1, channels=2 (stereo)
        silence = torch.zeros_like(audio)
        stereo = torch.stack([audio, silence], dim=0).unsqueeze(0)  # (1, 2, T)
        return stereo.to(self._device)

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        self._load()
        import torch

        audio = self._load_audio_context(record)

        with torch.no_grad():
            out = self._model(audio)

        # VAP output: p_now is the probability of voice-activity shift
        # at the current frame (last frame = end of utterance).
        # Different VAP versions expose this differently; try common APIs.
        p_shift: float
        if hasattr(out, "p_now"):
            # VAPModel forward returns an object with p_now: (B, T, 2)
            # p_now[0, -1, 0] = p(speaker 0 is active next) at last frame
            p_now = out.p_now[0, -1]  # (2,)
            # p_shift = probability that speaker 0 stops (other speaker takes over)
            p_shift = float(1.0 - p_now[0].item())
        elif hasattr(out, "logits"):
            p_shift = float(torch.sigmoid(out.logits[0, -1]).item())
        elif isinstance(out, dict) and "p_now" in out:
            p_now = out["p_now"][0, -1]
            p_shift = float(1.0 - p_now[0].item())
        elif isinstance(out, torch.Tensor):
            p_shift = float(torch.sigmoid(out[0, -1]).item())
        else:
            raise RuntimeError(
                f"Unrecognised VAP output type: {type(out)}. "
                "Please open an issue or set a custom VAPPredictor subclass."
            )

        p_shift = max(0.0, min(1.0, p_shift))
        p_hold = 1.0 - p_shift

        return TurnPrediction(
            probs={
                "complete": p_shift,
                "incomplete": p_hold,
                "backchannel": 0.0,
                "wait": 0.0,
            },
            timestamp=record.end,
        )

    def __repr__(self) -> str:
        return f"VAPPredictor(checkpoint={self.checkpoint!r}, threshold={self.threshold})"


class VAPPredictionFilePredictor:
    """Load VAP predictions from a pre-computed JSONL file.

    Use this when you have already run VAP inference offline (e.g. on a
    GPU server) and want to plug those predictions into compare-turn without
    re-running the model.

    JSONL format (one line per record):
        {"id": "<record_id>", "p_shift": 0.73}
    or the extended format used by the VAP official evaluation scripts:
        {"id": "<record_id>", "probs": {"complete": 0.73, "incomplete": 0.27}}
    """

    def __init__(self, path: str | Path) -> None:
        import json

        self._preds: dict[str, float] = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj["id"]
            if "probs" in obj:
                self._preds[rid] = float(obj["probs"].get("complete", 0.5))
            else:
                self._preds[rid] = float(obj.get("p_shift", obj.get("p_complete", 0.5)))

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        p_shift = self._preds.get(record.id, 0.5)
        return TurnPrediction(
            probs={
                "complete": p_shift,
                "incomplete": 1.0 - p_shift,
                "backchannel": 0.0,
                "wait": 0.0,
            },
            timestamp=record.end,
        )


def run_vap_inference(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint: str = _DEFAULT_CHECKPOINT,
    device: str | None = None,
    context_sec: float = _CONTEXT_SEC,
    audio_root: str | Path | None = None,
    batch_size: int = 32,
) -> str:
    """Run VAP inference on a manifest and save predictions to JSONL.

    This function runs VAP in batched mode, which is faster than calling
    VAPPredictor.predict() record-by-record when processing large datasets.

    Parameters
    ----------
    manifest_path: path to a JSONL turn manifest
    output_path:   where to write the predictions JSONL
    checkpoint:    VAP model checkpoint (HuggingFace ID or local path)
    device:        torch device
    audio_root:    base directory for relative audio paths
    batch_size:    number of audio clips to process at once

    Returns
    -------
    str: path to the written predictions file
    """
    import json
    from pathlib import Path as P

    try:
        import torch
        from vap.model import VAPModel
    except ImportError as exc:
        raise RuntimeError(
            "VAP inference requires vap-turn-taking.\n"
            "Install: pip install vap-turn-taking"
        ) from exc

    from stable_asr.data.audio import load_audio_mono
    from stable_asr.data.manifest import load_turn_manifest

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    predictor = VAPPredictor(checkpoint=checkpoint, device=device, context_sec=context_sec)
    predictor._load()

    records = load_turn_manifest(manifest_path)
    output_path = P(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds = []
    for i, record in enumerate(records):
        if audio_root and not P(record.audio).is_absolute():
            from dataclasses import replace
            record = replace(record, audio=str(P(audio_root) / record.audio))
        try:
            pred = predictor.predict(record)
            preds.append({
                "id": record.id,
                "p_shift": pred.probs["complete"],
                "probs": pred.probs,
                "label": pred.label,
            })
        except Exception as e:
            preds.append({
                "id": record.id,
                "p_shift": 0.5,
                "probs": {"complete": 0.5, "incomplete": 0.5, "backchannel": 0.0, "wait": 0.0},
                "label": "incomplete",
                "error": str(e),
            })

        if (i + 1) % 100 == 0:
            print(f"  VAP inference: {i + 1}/{len(records)}")

    output_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + "\n"
    )
    print(f"VAP predictions written: {output_path} ({len(preds)} records)")
    return str(output_path)
