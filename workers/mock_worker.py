"""Mock worker for testing (CPU-only, generates silence)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logging import get_logger

log = get_logger("mock_worker")

app = FastAPI(title="Mock Worker")


class GenRequest(BaseModel):
    text: str
    out_path: str
    target: str = "english"
    language: str | None = None
    reference_wav: str | None = None
    reference_text: str | None = None
    preset: dict = {}
    emotion_tags: str | None = None
    int4: bool = True
    quantize_4bit: bool = True
    clone_mode: str = "hybrid"
    cue_index: int = 0


class GenResponse(BaseModel):
    ok: bool
    wav_path: str | None = None
    seconds: float = 0.0
    error: str | None = None
    trace: str | None = None
    skipped: bool = False


@app.get("/health")
async def health():
    return {"ok": True, "model": "mock", "loaded": True, "device": "cpu"}


@app.post("/load")
async def load():
    log.info("Mock worker load called")
    return {"ok": True}


@app.post("/unload")
async def unload():
    log.info("Mock worker unload called")
    return {"ok": True}


@app.post("/generate")
async def generate(req: GenRequest):
    """Generate mock audio (silence)."""
    log.info("Generating mock audio for: %s...", req.text[:50])

    # Simulate processing time
    await asyncio.sleep(0.1)

    # Generate silence based on text length (~3 words/sec)
    words = len(req.text.split())
    duration = max(1.0, words / 3.0)
    sr = 48000
    samples = int(duration * sr)

    # Create silence
    audio = np.zeros(samples, dtype=np.float32)

    # Ensure output directory exists
    Path(req.out_path).parent.mkdir(parents=True, exist_ok=True)

    # Write WAV
    sf.write(req.out_path, audio, sr)

    return GenResponse(ok=True, wav_path=req.out_path, seconds=duration)


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "8100"))
    log.info("Starting mock worker on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
