"""VibeVoice-Hindi worker (runs in 'vibevoice' venv, Python 3.11).

Uses the community VibeVoice pipeline with the Hindi finetune (tarun7r/vibevoice-hindi-1.5B
by default; 7B/4-bit optional). transformers pinned to 4.51.3 in this venv.
flash-attn optional -> falls back to SDPA automatically (forced sdpa to avoid hard fails).
4-bit quantization used to fit 24 GB.

NOTE: The community VibeVoice inference entrypoint varies by fork. This worker uses the
documented `VibeVoiceForConditionalGeneration` + processor pattern and degrades gracefully
with a clear error if the installed fork differs, so the app never crashes — it just marks
this worker unavailable and offers a fallback model.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_worker import BaseWorker, run_worker  # noqa: E402
from protocol import GenRequest  # noqa: E402

DEFAULT_MODEL = os.environ.get("VIBEVOICE_MODEL", "tarun7r/vibevoice-hindi-1.5B")


class VibeVoiceWorker(BaseWorker):
    model_id = "vibevoice"

    def load_model(self):
        import torch
        # Avoid flash-attn hard requirement: force sdpa.
        os.environ.setdefault("VIBEVOICE_ATTN", "sdpa")
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference as VV,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        except Exception as e:
            raise RuntimeError(
                "VibeVoice community package not importable in this venv. "
                "Run scripts/install_model_vibevoice.sh. "
                f"Underlying import error: {e}"
            )
        # T4/Turing has no BF16 tensor support.  Loading a BF16 Hindi checkpoint
        # or using BF16 as bitsandbytes compute dtype causes an avoidable failure;
        # use FP16 there and BF16 only on Ampere/Ada/Hopper.
        dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)
        quant = None
        if os.environ.get("VIBEVOICE_4BIT", "1") == "1":
            try:
                from transformers import BitsAndBytesConfig
                quant = BitsAndBytesConfig(load_in_4bit=True,
                                           bnb_4bit_compute_dtype=dtype)
            except Exception:
                quant = None
        self._processor = VibeVoiceProcessor.from_pretrained(DEFAULT_MODEL)
        self._model = VV.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            attn_implementation="sdpa",
            quantization_config=quant,
        )
        self._model.eval()

    def synthesize(self, req: GenRequest) -> float:
        import soundfile as sf
        import numpy as np
        # VibeVoice expects a script and one or more speaker voice samples.
        voices = [req.reference_wav] if req.reference_wav else None
        # The community VibeVoice parser requires an explicit speaker label.
        # Passing bare narration makes it emit "Could not parse line" and produce
        # no usable speech, as seen in the worker log.
        script = f"Speaker 0: {(req.text or '').strip()}"
        inputs = self._processor(
            text=[script],
            voice_samples=[voices] if voices else None,
            padding=True,
            return_tensors="pt",
        )
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, tokenizer=self._processor.tokenizer)
        # community pipeline returns audio under .speech_outputs or similar
        audio = getattr(out, "speech_outputs", None)
        if audio is None:
            audio = out
        arr = audio[0]
        if hasattr(arr, "detach"):
            arr = arr.detach().float().cpu().numpy()
        arr = np.asarray(arr, dtype="float32").reshape(-1)
        sr = 24000
        sf.write(req.out_path, arr, sr)
        return float(len(arr)) / float(sr)


if __name__ == "__main__":
    run_worker(VibeVoiceWorker())
