"""Base HTTP worker using only Python stdlib (http.server) so it runs in ANY venv.

Each concrete worker subclasses BaseWorker and implements:
    load_model(self)       -> loads weights into VRAM (download on first call)
    unload_model(self)     -> frees VRAM
    synthesize(self, req)  -> writes req.out_path, returns duration seconds

Run a worker:  python -m worker_xxx --port 8101
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# protocol lives beside the worker files; add cwd for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import GenRequest  # noqa: E402


class BaseWorker:
    model_id = "base"

    def __init__(self):
        self._loaded = False
        self._model = None

    # ---- to implement in subclasses ----
    def load_model(self):
        raise NotImplementedError

    def unload_model(self):
        self._model = None
        self._loaded = False

    def synthesize(self, req: GenRequest) -> float:
        raise NotImplementedError

    # ---- long-cue split-and-rejoin (only for very long cues) ----
    def generate_dispatch(self, req: GenRequest) -> float:
        """Called by the HTTP handler. If the cue text is very long (>= SPLIT_CHARS,
        default 600), split it at sentence boundaries, synthesize each chunk, and
        rejoin with a short crossfade into ONE cue WAV (timeline stays 1 cue=1 seg).
        Short/normal cues are synthesized directly — untouched, zero quality change."""
        import os
        threshold = int(os.environ.get("TTS_SPLIT_CHARS", "600"))
        split_on = os.environ.get("TTS_SPLIT", "1") == "1"
        text = req.text or ""
        if (not split_on) or len(text) < threshold:
            return self.synthesize(req)   # normal path, unchanged

        # split at sentence/clause boundaries only (never mid-word)
        import re
        import wave
        import struct
        parts = re.split(r"(?<=[।.!?])\s+", text.strip())
        # merge tiny fragments up to ~threshold so we don't over-split
        chunks, cur = [], ""
        for p in parts:
            if not p:
                continue
            if len(cur) + len(p) + 1 <= threshold:
                cur = (cur + " " + p).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        if len(chunks) <= 1:
            return self.synthesize(req)   # couldn't split safely -> just do it whole

        # synthesize each chunk to a temp wav
        import dataclasses
        tmp_paths = []
        base = req.out_path
        for i, ch in enumerate(chunks):
            tp = f"{base}.part{i}.wav"
            sub = dataclasses.replace(req, text=ch, out_path=tp)
            self.synthesize(sub)
            tmp_paths.append(tp)

        # rejoin with 15ms crossfade using soundfile + the project's join logic
        try:
            import numpy as np
            import soundfile as sf
            arrs, sr = [], None
            for tp in tmp_paths:
                a, s = sf.read(tp)
                a = np.asarray(a, dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                arrs.append(a); sr = s
            n = int((sr or 24000) * 15 / 1000)   # 15 ms crossfade
            out = arrs[0]
            for nxt in arrs[1:]:
                if n > 0 and len(out) >= n and len(nxt) >= n:
                    t = np.linspace(0, np.pi / 2, n)
                    fout = np.cos(t) ** 2
                    fin = np.sin(t) ** 2
                    out = np.concatenate([out[:-n], out[-n:] * fout + nxt[:n] * fin, nxt[n:]])
                else:
                    out = np.concatenate([out, nxt])
            sf.write(base, out.astype("float32"), sr or 24000)
            dur = float(len(out)) / float(sr or 24000)
        finally:
            for tp in tmp_paths:
                try:
                    os.remove(tp)
                except Exception:
                    pass
        return dur

    # ---- helpers ----
    @property
    def loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self):
        if not self._loaded:
            self.load_model()
            self._loaded = True

    def device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def _make_handler(worker: BaseWorker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": worker.model_id,
                                 "loaded": worker.loaded, "device": worker.device()})
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            try:
                if self.path == "/load":
                    worker.ensure_loaded()
                    self._send(200, {"ok": True, "loaded": True})
                elif self.path == "/unload":
                    worker.unload_model()
                    try:
                        import torch, gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    self._send(200, {"ok": True, "loaded": False})
                elif self.path == "/generate":
                    req = GenRequest.from_json(self._read())
                    worker.ensure_loaded()
                    secs = worker.generate_dispatch(req)
                    self._send(200, {"ok": True, "wav_path": req.out_path, "seconds": secs})
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except Exception as e:
                self._send(500, {"ok": False, "error": str(e),
                                 "trace": traceback.format_exc()})
    return Handler


def run_worker(worker: BaseWorker):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(worker))
    print(f"[worker:{worker.model_id}] listening on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
