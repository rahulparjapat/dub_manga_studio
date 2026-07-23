# Chatterbox Manga Studio — Multi-Model Edition

Convert Chinese manga-explainer videos into **English / Hindi / Hinglish** dubbed
videos with AI script adaptation, **5 selectable TTS models**, cue-locked retiming,
subtitle masking, BGM, and YouTube-ready MP4 export.

**Built for:** Lightning AI Studio · Linux · single 24 GB GPU (A10G / L4).

## ✨ Channel & quality features
- **AI adaptation:** strict per-cue JSON alignment, duration-fit, cross-batch context
  carryover, auto-updating glossary, back-translation quality check, optimized prompts.
- **🎬 Retention presets** (Tab 2): Cliffhanger · Fast Recap · Deep Lore · Reaction/Hype ·
  Chill Explain — set tone & pacing, language-agnostic (works for EN/Hindi/Hinglish).
- **💾 Full-setup presets** (Tab 2): save your whole setup (target + style + retention +
  emotions) and load it in one click per new video.
- **Live progress** on transcribe / dub / Auto pipeline, responsive UI (no freeze).
- **🛑 Cancel current dub** — stop a running dub safely; finished cues are kept and
  skipped on the next run (resume).
- **BGM with ducking sliders** (level + duck depth), sidechain compression, loudnorm.
- **Frame-accurate export** with keyframe-gated stream-copy fast-path (needs ffprobe).
- **Direct Text → Audio** with optional AI-adapt (Tab 4).
- **Live Logs tab** (Tab 7): watch model downloads, venv setup, transcription,
  dubbing (cue-by-cue) and export in real time; auto-refresh + download.

## ⚠️ VoxCPM2 / IndicF5 need a reference voice
VoxCPM2 and IndicF5 clone a **reference voice**. In Tab 3 set *Voice source =
"Saved reference voice"*, pick a saved `.wav`, and fill *Reference transcript*
(what that clip says). Without a reference the app now blocks with a clear message
(previously it produced a random/inconsistent voice). Emotion `(style)` tags are
kept for VoxCPM2/Fish (they read them) and auto-stripped for models that can't
(so tags are never spoken aloud).

## 💾 Disk-frugal mode (10 GB total budget)
This project is tuned to run inside a **10 GB total disk** (Lightning free tier):

- The **app venv is tiny** (Gradio only, **no torch** ≈ 0.5 GB).
- **Whisper stays CACHED on disk permanently** (faster-whisper = CTranslate2, **no torch**, ~3 GB with INT8). It installs once and is **never disk-evicted** — it only unloads from **VRAM** after each transcription. So you never re-download it.
- **Each TTS model installs ON DEMAND** (first Dub) and other TTS models are **auto-evicted** so only **one TTS model lives on disk at a time** (Whisper is protected and kept).
- After a dub finishes, that TTS model's weights are **cleared automatically** (toggle in Tab 3).
- **Budget for one TTS model ≈ 7 GB** (10 GB − ~3 GB reserved for cached Whisper).
- **Fits alongside Whisper:** IndicF5 (~7 GB) ✅, Chatterbox (~7 GB) ✅.
- **Too big for 10 GB with Whisper cached:** VibeVoice (~9 GB), VoxCPM2 (~10 GB), **Fish S2 Pro (~15 GB)** — use cloud API or a bigger disk.

Tab 6 → *Storage & Cleanup* shows live disk, installed models, and lets you evict any model.

---

## Why 5 models (and why isolated environments)

Your Chatterbox Hinglish sounded slightly robotic / non-native. So this edition lets
you pick the best engine per job:

| Model | Best for | License | Load VRAM | Voice input |
|---|---|---|---|---|
| **IndicF5** ⭐ default for Hindi/Hinglish | Native Hindi/Hinglish (1,417h Indian speech) | AI4Bharat gated — verify commercial | ~4 GB | ref + transcript |
| **VoxCPM2** | Native + **free commercial** (Apache-2.0) | Apache 2.0 | ~8 GB | ref + transcript (optional) |
| **Chatterbox** default for English | English, easy, MIT | MIT | ~6 GB | ref |
| **VibeVoice-Hindi** | Long-form Hindi | Community — verify | ~12 GB (4-bit) | ref |
| **Fish S2 Pro** | Top quality + inline emotion tags | ⚠ **PAID for commercial self-host** | ~12 GB (int4) | ref + `[tags]` |

> These 5 have **incompatible Python dependencies** (verified: Chatterbox needs
> transformers 4.46, IndicF5 <4.50, VibeVoice **exactly 4.51.3**, Fish its own stack).
> So each runs in its **own venv** as a small local worker. The app loads **one model
> at a time**, downloads weights **only when you click Dub**, and **unloads after**.

### ⚠ License honesty
- **Fish S2 Pro**: free for research/non-commercial only. **Monetized/commercial
  self-host needs a PAID license from Fish Audio.** Surfaced in the UI.
- **IndicF5 / VibeVoice-Hindi**: confirm their licenses before monetized use.
- **Chatterbox (MIT) / VoxCPM2 (Apache-2.0)**: free for commercial use.

---

## Setup on Lightning (one time)

```bash
cd ~/chatterbox_manga_studio

# 1) Put your HuggingFace token (needed for IndicF5 & Fish gated weights)
echo "hf_XXXXXXXXXXXX" > hf_token.txt        # or paste it in the Settings tab later

# 2) Bootstrap: system pkgs + app env + all 5 model venvs (weights NOT downloaded yet)
bash scripts/bootstrap_lightning.sh

# 3) Check environment
source .venv_app/bin/activate
python scripts/check_environment.py

# 4) Run
python app.py
# open the printed gradio.live link
```

**Model weights download on first Dub** (one-time, with a size warning — Fish is ~9 GB).
Nothing heavy downloads at app start, so the app boots instantly.

### GPU selection
Edit `active_gpu:` in `config.yaml` to match your Lightning GPU
(`a10g`, `l4`, `a100_40`, …). Switch GPU on Lightning → set this → restart.
`check_environment.py` prints the right value.

---

## Using it (workflow — nothing skipped)

1. **Tab 1 — Ingest & Transcribe:** upload / input-folder / Drive; transcribe with
   faster-whisper large-v3 (float16, VAD, word timestamps, batch 16→8→4).
2. **Tab 2 — Script & Adaptation:** 4 targets; manual / import SRT / AI adapt
   (Gemini/Groq/OpenRouter/Cerebras) with model browser, prompt layers, project
   templates, 6-batch manager (retry/restore/versions/pause), auto-glossary,
   Devanagari-preferred.
3. **Tab 3 — Dubbing:** pick one of the **5 models**; forward package + override
   versions; built-in or cloned voice; energy preset; cue cleanup; live-render
   pipeline; **lazy download + unload after**.
4. **Tab 4 — Direct Text to Audio:** standalone WAV/MP3, optional AI adapt.
5. **Tab 5 — Subtitles & Export:** presets; cue-locked + other timing modes;
   captions (derive/AI/import); Chinese subtitle mask (blur/band/pixelate/cover +
   chroma safety); BGM + ducking + loudnorm + limiter; silence compression;
   YouTube metadata (multi-lang, TXT/JSON/CSV); MP4 + SRT + script + quality report;
   fast copy export.
   *Export note:* video retiming is **frame-accurate** (setpts on `out/src`).
   Unchanged segments that land on keyframes are **stream-copied** (lossless, faster)
   — this needs **ffprobe** (bundled with the `ffmpeg` apt package). If ffprobe is
   absent the export still works and stays exact; it just re-encodes every segment.
6. **Tab 6 — Settings:** Prompt Studio, provider keys + HF token, 90-min keepalive,
   storage cleanup.

### Reference voice (for native cloning)
```bash
python scripts/make_reference_voice.py my_clip.mp3 data/voices/my_voice.wav --seconds 8
# then edit data/voices/my_voice.txt with the EXACT words spoken (IndicF5/VoxCPM2 need it)
```

### Benchmark a model on your GPU
```bash
python scripts/benchmark_tts.py --model indicf5 --target hinglish_devanagari \
  --ref data/voices/my_voice.wav --ref-text "$(cat data/voices/my_voice.txt)"
```

### Prove the real models work (GPU self-test)
After bootstrap, confirm each installed model actually produces audio:
```bash
python scripts/selftest_models.py            # tests all installed models
python scripts/selftest_models.py --model indicf5
```
Real WAVs land in `data/output/selftest/`. This is the check that can only run on
the GPU (weights download on first run).

---

## Security
- HuggingFace token & provider keys are stored in **local files** (`hf_token.txt`,
  `provider_keys.json`) that are **git-ignored** and never written into exports.
- **Never commit these files.** If a token leaks, revoke it at
  https://huggingface.co/settings/tokens and create a new one.

## What was verified vs. what runs on your GPU
- Verified in build: all code compiles, the full unit-test suite passes, the whole
  3-group / 6-tab Gradio UI builds, the worker HTTP contract (load/generate/unload)
  works, and installer syntax is valid.
- On your Studio: actual model weight downloads + first real GPU generation + audio
  quality. Installers use source-verified commands and are self-checking; a model that
  fails to install is hidden from the dropdown so the app still runs with the others.

## Keeping your Lightning Studio awake (honest note)

The app has a built-in keep-alive (auto-on, 90 min; `app.keepalive` /
`app.keepalive_minutes` in `config.yaml`) that pings the local Gradio server so
the session and the warm Whisper worker stay active.

**Important:** a local ping CANNOT override Lightning AI's own Studio idle/sleep
(billing) timeout — only Lightning controls that. To stop the Studio itself from
sleeping during long runs, raise or disable the **auto-timeout / idle-shutdown**
setting in Lightning's own UI (check the current label in the live UI; it varies
by plan). Running an actual job (real GPU/CPU work) resets Lightning's idle timer
far more reliably than any ping.
# dub_manga_studio
