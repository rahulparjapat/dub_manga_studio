"""Shared worker <-> app protocol (pure stdlib so it imports in ANY venv).

Every model worker is a small HTTP server exposing:
  GET  /health   -> {"ok": true, "model": "...", "loaded": bool, "device": "cuda"}
  POST /load     -> {"ok": true}                     (download weights if needed + load to VRAM)
  POST /generate -> {"ok": true, "wav_path": "...", "seconds": float}
  POST /unload   -> {"ok": true}                     (free VRAM)

Request for /generate (JSON):
{
  "text": "...",                 # target narration text
  "language": "hi|en|...",       # language id
  "target": "hinglish_roman",    # our target id
  "out_path": "/abs/out.wav",
  "reference_wav": "/abs/ref.wav" | null,
  "reference_text": "transcript of ref" | null,
  "preset": {"exaggeration":..,"cfg_weight":..,"temperature":..,
             "repetition_penalty":..,"cfg_value":..,"inference_timesteps":..},
  "emotion_tags": "[excited]" | null,   # Fish inline tags
  "int4": true|false,                    # Fish quantization
  "quantize_4bit": true|false,           # VibeVoice
  "clone_mode": "hybrid|hifi|controllable",  # VoxCPM2 clone mode
  "cue_index": 0                          # cue index for hybrid routing
}
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional


# ---- target -> (language_id) map used by workers ----
TARGET_LANG = {
    "english": "en",
    "hindi_devanagari": "hi",
    "hinglish_roman": "hi",
    "hinglish_devanagari": "hi",
}


@dataclass
class GenRequest:
    text: str
    out_path: str
    target: str = "english"
    language: Optional[str] = None      # None -> derived from target
    reference_wav: Optional[str] = None
    reference_text: Optional[str] = None
    preset: dict = field(default_factory=dict)
    emotion_tags: Optional[str] = None
    int4: bool = True
    quantize_4bit: bool = True
    clone_mode: str = "hybrid"          # VoxCPM2: "hifi" | "controllable" | "hybrid"
    cue_index: int = 0                  # cue index for hybrid routing

    def __post_init__(self):
        if not self.language:
            self.language = TARGET_LANG.get(self.target, "en")

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "GenRequest":
        return cls(
            text=d.get("text", ""),
            out_path=d["out_path"],
            target=d.get("target", "english"),
            language=d.get("language"),
            reference_wav=d.get("reference_wav"),
            reference_text=d.get("reference_text"),
            preset=d.get("preset") or {},
            emotion_tags=d.get("emotion_tags"),
            int4=bool(d.get("int4", True)),
            quantize_4bit=bool(d.get("quantize_4bit", True)),
            clone_mode=d.get("clone_mode", "hybrid"),
            cue_index=d.get("cue_index", 0),
        )
