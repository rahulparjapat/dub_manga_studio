"""IndicF5 worker (runs in 'indicf5' venv, Python 3.10) — PRIMARY Hinglish engine.

Verified: transformers.AutoModel, repo_id="ai4bharat/IndicF5", trust_remote_code=True.
Inputs required: text + reference audio + reference transcript.
Gated model -> needs HF token in env (handled by app before launch).

CRITICAL Roman-Hinglish handling (verified from harrrshall/hinglish-tts):
- The unpatched model truncates Roman-script output.
- We convert Roman Hinglish -> Devanagari via ai4bharat-transliteration when the
  target is Roman/Devanagari-preferred so pronunciation is native and complete.
- If transliteration lib is missing, we fall back to raw text with a warning.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_worker import BaseWorker, run_worker  # noqa: E402
from protocol import GenRequest  # noqa: E402

REPO_ID = "ai4bharat/IndicF5"


class IndicF5Worker(BaseWorker):
    model_id = "indicf5"

    def load_model(self):
        import torch
        from transformers import AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model = AutoModel.from_pretrained(REPO_ID, trust_remote_code=True)
        self._model = self._model.to(device)
        # transliterator (Roman -> Devanagari) for Hinglish
        self._xlit = None
        try:
            from ai4bharat.transliteration import XlitEngine
            self._xlit = XlitEngine("hi", beam_width=10, rescore=False)
        except Exception as e:
            print(f"[indicf5] transliteration unavailable ({e}); "
                  f"Roman input used as-is.", flush=True)

    def _prep_text(self, req: GenRequest) -> str:
        text = req.text
        # For Roman Hinglish / Devanagari-preferred, convert latin words to Devanagari
        if req.target in ("hinglish_roman", "hinglish_devanagari") and self._xlit:
            out_words = []
            for w in text.split():
                # transliterate tokens that are latin alphabetic
                core = "".join(ch for ch in w if ch.isalpha())
                if core.isascii() and core.isalpha():
                    try:
                        tr = self._xlit.translit_word(core, topk=1)
                        rep = tr[0] if isinstance(tr, list) and tr else w
                        out_words.append(w.replace(core, rep))
                    except Exception:
                        out_words.append(w)
                else:
                    out_words.append(w)
            text = " ".join(out_words)
        return text

    def synthesize(self, req: GenRequest) -> float:
        import soundfile as sf
        if not req.reference_wav or not req.reference_text:
            raise ValueError("IndicF5 requires a reference audio + its transcript. "
                             "Provide a saved reference voice with transcript.")
        text = self._prep_text(req)
        audio = self._model(text, ref_audio_path=req.reference_wav, ref_text=req.reference_text)
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sr = 24000
        sf.write(req.out_path, audio.astype(np.float32), samplerate=sr)
        return float(len(audio)) / float(sr)


if __name__ == "__main__":
    run_worker(IndicF5Worker())
