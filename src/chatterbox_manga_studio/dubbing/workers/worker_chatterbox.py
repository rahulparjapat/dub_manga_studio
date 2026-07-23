"""Chatterbox worker (runs in the 'chatterbox' venv, Python 3.10).

Verified API: chatterbox.mtl_tts.ChatterboxMultilingualTTS / chatterbox.tts.ChatterboxTTS
Generation params: exaggeration, cfg_weight, temperature, repetition_penalty, language_id.

Precision: bf16 ONLY on GPUs that support it (checked at runtime via
torch.cuda.is_bf16_supported()). On T4 (Turing, no bf16) it stays fp16/default —
so the SAME code runs on both T4 and L4 with no crash.

M3 (watermark honesty): Chatterbox's own library applies Resemble PerTh watermarking
internally to its generated audio; we do not add or remove it here. The config
'watermark_default' flag is informational (Chatterbox = on by design; other models
have no built-in watermark).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_worker import BaseWorker, run_worker  # noqa: E402
from protocol import GenRequest  # noqa: E402


class ChatterboxWorker(BaseWorker):
    model_id = "chatterbox"

    def load_model(self):
        import torch

        self._multilingual = True
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as CB
        except Exception:
            from chatterbox.tts import ChatterboxTTS as CB

            self._multilingual = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CB.from_pretrained(device=device)

        bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
        # bf16 on capable GPUs (L4/A10G/A100). T4 (no bf16) stays fp16/default.
        if bf16_ok:
            try:
                self._model.t3.to(dtype=torch.bfloat16)
                self._model.conds.t3.to(dtype=torch.bfloat16)
            except Exception as e:
                print(f"[chatterbox] bf16 apply skipped: {e}", flush=True)

        # torch.compile of the T3 step (verified 2-3x on capable GPUs).
        # Gated by env TTS_COMPILE (router sets it only when the GPU profile allows,
        # i.e. NOT on T4). Auto-off + safe fallback if anything fails.
        want_compile = os.environ.get("TTS_COMPILE", "0") == "1"
        if want_compile and bf16_ok:
            try:
                tgt = getattr(self._model.t3, "_step_compilation_target", None)
                if tgt is not None:
                    self._model.t3._step_compilation_target = torch.compile(
                        tgt, fullgraph=True, backend="cudagraphs"
                    )
                    print("[chatterbox] torch.compile enabled (cudagraphs).", flush=True)
            except Exception as e:
                print(f"[chatterbox] torch.compile skipped ({e}); running uncompiled.", flush=True)

    def synthesize(self, req: GenRequest) -> float:
        import torchaudio as ta

        p = req.preset or {}
        kw = {
            "exaggeration": p.get("exaggeration", 0.5),
            "cfg_weight": p.get("cfg_weight", 0.5),
            "temperature": p.get("temperature", 0.8),
            "repetition_penalty": p.get("repetition_penalty", 2.0),
        }
        if getattr(self, "_multilingual", False):
            kw["language_id"] = req.language or "en"
        if req.reference_wav:
            kw["audio_prompt_path"] = req.reference_wav
            # cross-language accent fix: ref present but target != en -> cfg 0
            if (req.language or "en") != "en":
                kw["cfg_weight"] = 0.0
        wav = self._model.generate(req.text, **kw)
        ta.save(req.out_path, wav, self._model.sr)
        return float(wav.shape[-1]) / float(self._model.sr)


if __name__ == "__main__":
    run_worker(ChatterboxWorker())
