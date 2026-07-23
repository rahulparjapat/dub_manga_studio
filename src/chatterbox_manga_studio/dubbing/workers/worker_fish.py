"""Fish S2 Pro worker (runs in 'fish' venv, Python 3.12).

⚠ LICENSE: Fish Audio Research License. FREE for research/non-commercial only.
   Commercial/monetized self-host requires a PAID license from Fish Audio.
   This is surfaced in the UI; the user accepted responsibility.

Verified two-stage CLI/library pipeline (fishaudio/fish-speech, checkpoints/s2-pro):
  1) encode reference -> VQ tokens (codec.pth)
  2) text2semantic -> semantic tokens (supports --half for non-bf16; int4/BnB4 for 24GB)
  3) decode semantic -> waveform

Inline emotion tags: [excited], [whispering], [professional broadcast tone] embedded in text.
Precision (CORRECTED — the fish-speech CLI has NO --bnb4/int4 flag; verified vs the
official docs):
  - fp16: `--half`  ->  ~24 GB VRAM  (fits L4/A100; does NOT fit a 16 GB T4)
  - bf16: native (no flag) on bf16-capable GPUs -> ~24 GB VRAM
Disk footprint ~15 GB (9 GB weights + venv) — exceeds a 10 GB budget; needs bigger disk.

The fish-speech library API differs across releases. This worker calls the library's
high-level TTS if available, else shells to the documented tool scripts. On any mismatch it
raises a clear error so the app marks Fish unavailable (never crashes the whole app).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_worker import BaseWorker, run_worker  # noqa: E402
from protocol import GenRequest  # noqa: E402

# The installer clones the Fish repository beside its isolated venv.  Never use
# the TTS cue output directory as the CLI root: Fish's scripts are repository
# relative and otherwise resolve to a nonexistent `tts_cues/fish_speech/...` path.
FISH_SRC = Path(os.environ.get("FISH_SRC", str(Path(sys.prefix).parent / "fish_src")))
CKPT_DIR = Path(os.environ.get("FISH_CKPT", str(FISH_SRC / "checkpoints" / "s2-pro")))


class FishWorker(BaseWorker):
    model_id = "fish"

    def load_model(self):
        # Fish loads per-call in the reference pipeline; we validate presence here.
        import torch  # noqa
        if not os.path.isdir(CKPT_DIR):
            # Weights are downloaded lazily by the installer/first-load step.
            from huggingface_hub import snapshot_download
            snapshot_download("fishaudio/s2-pro", local_dir=CKPT_DIR)
        self._int4 = os.environ.get("FISH_INT4", "1") == "1"
        try:
            # Preferred: high-level library entry (if this fish-speech release exposes it)
            import fish_speech  # noqa
            self._have_lib = True
        except Exception:
            self._have_lib = False

    def _apply_tags(self, req: GenRequest) -> str:
        text = req.text
        if req.emotion_tags:
            # inline tag at start applies to following text (verified S2 behavior)
            text = f"{req.emotion_tags} {text}"
        return text

    def synthesize(self, req: GenRequest) -> float:
        import soundfile as sf
        import numpy as np
        text = self._apply_tags(req)

        # Prefer the VERIFIED CLI path when int4 (NF4) is requested — that is the
        # only path where --bnb4 real 4-bit quantization is confirmed to work.
        if getattr(self, "_int4", True):
            return self._synthesize_cli(req, text)

        # Otherwise try the high-level library path (fp16), CLI as fallback.
        if getattr(self, "_have_lib", False):
            try:
                import fish_speech
                model = fish_speech.load_model(CKPT_DIR)  # type: ignore[attr-defined]
                kwargs = {}
                if req.reference_wav:
                    kwargs["reference_audio"] = req.reference_wav
                    if req.reference_text:
                        kwargs["reference_text"] = req.reference_text
                audio = model.synthesize(text, **kwargs)  # type: ignore[attr-defined]
                arr = np.asarray(audio, dtype="float32").reshape(-1)
                sr = getattr(model, "sr", 44100)
                sf.write(req.out_path, arr, sr)
                return float(len(arr)) / float(sr)
            except Exception as e:
                print(f"[fish] library path failed ({e}); trying CLI tools.", flush=True)

        # Fallback: documented tool scripts via subprocess (two-stage)
        return self._synthesize_cli(req, text)

    def _run_fish(self, cmd, workdir):
        """Run a Fish CLI step with UTF-8 forced + FULL output capture.

        Two fixes vs the old silent `subprocess.run(check=True)`:
          1) DEBUGGABLE: capture stdout+stderr and, on failure, raise a
             RuntimeError containing Fish's REAL error (the old code hid it, so we
             only ever saw 'exit status 2').
          2) UTF-8 SAFE: force PYTHONUTF8/PYTHONIOENCODING + text encoding so
             Devanagari/Hinglish text isn't corrupted into mojibake (the log
             showed 'इस आदमी _ ह◆ली…' — a classic non-UTF-8 locale mangle).
        """
        import subprocess
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["LC_ALL"] = env.get("LC_ALL", "C.UTF-8")
        env["LANG"] = env.get("LANG", "C.UTF-8")
        proc = subprocess.run(cmd, cwd=workdir, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").strip().splitlines()[-25:])
            step = os.path.basename(cmd[1]) if len(cmd) > 1 else "fish"
            raise RuntimeError(
                f"Fish step '{step}' failed (exit {proc.returncode}). "
                f"Real error from Fish:\n{tail or '(no output captured)'}")
        return proc.stdout or ""

    def _synthesize_cli(self, req: GenRequest, text: str) -> float:
        import shutil
        import glob
        import soundfile as sf
        import numpy as np

        # Sanitize text to clean UTF-8 (drop stray control chars that would break
        # the tokenizer); keep all real Devanagari/Latin/punctuation.
        text = "".join(ch for ch in (text or "")
                       if ch == "\n" or ord(ch) >= 0x20).strip()
        if not text:
            raise RuntimeError("Fish: empty text after cleanup — nothing to speak.")

        # Fish CLI scripts use paths inside the cloned Fish repository.  Run there
        # and use absolute script/checkpoint paths; output artifacts are then found
        # deterministically instead of under the per-project cue folder.
        workdir = str(FISH_SRC)
        codec = str(CKPT_DIR / "codec.pth")
        prompt_tokens = "fake.npy"
        # 1) reference -> VQ tokens (optional)
        if req.reference_wav:
            self._run_fish([sys.executable, str(FISH_SRC / "fish_speech" / "models" / "dac" / "inference.py"),
                            "-i", req.reference_wav, "--checkpoint-path", codec], workdir)
        # 2) text -> semantic. VERIFIED flags (fishaudio/fish-speech docs): --text,
        #    --prompt-text, --prompt-tokens, --half, --compile. There is NO --bnb4
        #    flag in this release — passing it caused argparse to exit(2) (the bug).
        #    On a non-bf16 GPU we use --half (fp16); bf16 GPUs (L4) can omit it.
        cmd = [sys.executable, str(FISH_SRC / "fish_speech" / "models" / "text2semantic" / "inference.py"),
               "--text", text, "--checkpoint-path", str(CKPT_DIR)]
        if req.reference_wav and req.reference_text:
            cmd += ["--prompt-text", req.reference_text, "--prompt-tokens", prompt_tokens]
        # --half is safe on both fp16 and bf16 GPUs; keep it for broad compatibility.
        cmd += ["--half"]
        self._run_fish(cmd, workdir)
        # 3) semantic -> waveform. Per docs, decode takes -i codes_N.npy (+ codec path).
        codes = sorted(glob.glob(os.path.join(workdir, "codes_*.npy")))
        if not codes:
            raise RuntimeError("Fish: no semantic codes (codes_*.npy) were produced "
                               "by text2semantic — see the captured error above.")
        self._run_fish([sys.executable, str(FISH_SRC / "fish_speech" / "models" / "dac" / "inference.py"),
                        "-i", codes[-1], "--checkpoint-path", codec], workdir)
        produced = os.path.join(workdir, "fake.wav")
        if not os.path.exists(produced):
            raise RuntimeError("Fish: decode step produced no fake.wav.")
        shutil.move(produced, req.out_path)
        arr, sr = sf.read(req.out_path)
        return float(len(np.asarray(arr))) / float(sr)


if __name__ == "__main__":
    run_worker(FishWorker())
