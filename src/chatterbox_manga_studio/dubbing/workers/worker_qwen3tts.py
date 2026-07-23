"""Qwen3-TTS worker (runs in 'qwen3tts' venv, Python 3.12).

Verified API (github.com/QwenLM/Qwen3-TTS, huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base):
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                dtype=torch.bfloat16|torch.float16, attn_implementation="flash_attention_2")
  - Voice CLONE (Base):  model.generate_clone(text, ref_audio=..., ref_text=...)
    (falls back to generate_custom_voice for the CustomVoice checkpoint)
  - Emotion via natural-language `instruct=` (e.g. "Very happy.").

FP16/BF16 ONLY: FlashAttention 2 requires fp16/bf16, so there is no int8/int4 path.
On T4 (no bf16) we use float16; on L4 we use bfloat16. Apache-2.0 -> commercial-safe.
10 languages (ZH/EN/JP/KR/DE/FR/RU/PT/ES/IT) — NOT trained on Hindi, so best for
English / Hinglish-Roman where English phonetics dominate.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_worker import BaseWorker, run_worker  # noqa: E402
from protocol import GenRequest  # noqa: E402

# Voice-clone checkpoint (needs ref audio + transcript, like VoxCPM2/IndicF5).
REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class Qwen3TTSWorker(BaseWorker):
    model_id = "qwen3tts"

    def load_model(self):
        import torch

        # T4 (Turing) cannot do bf16 -> use fp16; L4/Ada -> bf16. TTS_PRECISION is
        # set by the router from the active GPU profile.
        prec = os.environ.get("TTS_PRECISION", "float16").lower()
        if prec == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        from qwen_tts import Qwen3TTSModel

        kw = {"device_map": "cuda:0" if torch.cuda.is_available() else "cpu", "dtype": dtype}
        # FlashAttention 2 greatly cuts VRAM but isn't always installable; try it,
        # fall back to the default attention implementation if unavailable.
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                REPO_ID, attn_implementation="flash_attention_2", **kw
            )
        except Exception as e:
            print(f"[qwen3tts] flash_attention_2 unavailable ({e}); default attention.", flush=True)
            self._model = Qwen3TTSModel.from_pretrained(REPO_ID, **kw)

    def _lang_name(self, target: str) -> str:
        # Qwen3-TTS expects language NAMES; 'Auto' lets it detect.
        return {"english": "English"}.get(target, "Auto")

    def synthesize(self, req: GenRequest) -> float:
        import numpy as np
        import soundfile as sf

        if not req.reference_wav:
            raise ValueError(
                "Qwen3-TTS (Base clone) needs a reference voice. Use 'Auto default "
                "voice' or a saved reference voice."
            )
        # emotion_tags are natural-language style instructions for Qwen (e.g. 'Very happy.')
        (req.emotion_tags or "").strip() or None
        lang = self._lang_name(req.target)

        # Official Qwen Base checkpoint API is generate_voice_clone().  The Base
        # model does not accept `instruct` in clone mode; that belongs to the
        # VoiceDesign/CustomVoice variants. Passing it made fresh installs fail
        # after model download with an unexpected-keyword error.
        fn = getattr(self._model, "generate_voice_clone", None)
        if fn is None:
            raise RuntimeError(
                "Installed qwen-tts has no generate_voice_clone API; "
                "reinstall with scripts/install_model_qwen3tts.sh."
            )
        try:
            wavs, sr = fn(
                text=req.text,
                ref_audio=req.reference_wav,
                ref_text=req.reference_text or "",
                language=lang,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Qwen3-TTS voice clone failed: {e}") from e

        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        wav = np.asarray(wav, dtype="float32")
        sr = int(sr or 24000)
        sf.write(req.out_path, wav, sr)
        return float(len(wav)) / float(sr)


if __name__ == "__main__":
    run_worker(Qwen3TTSWorker())
