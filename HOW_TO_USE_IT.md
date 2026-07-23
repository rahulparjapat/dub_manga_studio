# How To Use It — quick, non-technical

## First time
1. `bash scripts/bootstrap_lightning.sh`  (installs everything, ~15–30 min)
2. Put your HuggingFace token in `hf_token.txt` (or the Settings tab).
3. `source .venv_app/bin/activate && python app.py` → open the gradio link.

## Make one dubbed video
1. **Tab 1**: name the project, upload the Chinese video, click **Transcribe**.
2. **Tab 2**: choose target = *Hinglish — Devanagari Preferred*, add your API key in
   Tab 6 if using AI, **Create Plan** → **Translate All**. (Or paste manual script.)
3. **Tab 3**: **Load latest forwarded script**, pick model **IndicF5**, choose a
   reference voice (+ its transcript), set energy **expressive**, click **▶ Dub**.
   - First time with a model it downloads weights (one-time). It unloads after.
4. **Tab 5**: pick **YouTube Standard**, optionally add captions / hide Chinese subs /
   BGM, click **Quick Export MP4**. Download the MP4/SRT/script/quality report.

## Which model when
- **Native Hinglish/Hindi** → IndicF5 (default). Free-commercial alternative → VoxCPM2.
- **English** → Chatterbox.
- **Absolute top quality / emotion tags** → Fish S2 Pro (⚠ paid for commercial).
- **Very long single-voice narration** → VibeVoice-Hindi.

## Tips for the most NATIVE voice
- Use a **clean 5–10s Hindi reference clip** (`scripts/make_reference_voice.py`), no music.
- Fill in the **reference transcript** (IndicF5/VoxCPM2 need it).
- If your reference voice isn't Hindi, the app auto-reduces accent bleed.

## If a model won't load
- The app hides broken models and keeps the rest working.
- Re-run its installer, e.g. `bash scripts/install_model_indicf5.sh`.
- `python scripts/check_environment.py` shows which venvs are installed.
