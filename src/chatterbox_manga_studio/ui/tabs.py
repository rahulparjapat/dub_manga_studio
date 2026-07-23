"""All 6 tab builders. Every feature from the original workflow is present.

The UI wires callbacks to the engine modules. Model weights are downloaded lazily
on first Dub (progress + one-time size warning). Models unload after dub finishes.
"""
from __future__ import annotations
import json
from pathlib import Path

import gradio as gr

from ..common import config as C
from ..common import keys as K
from ..common import hf_token as HFT
from ..common.paths import (project_dir, edition_dir, VOICES, BGM, INPUT,
                            DIRECT_AUDIO, safe_name, find_source_video)
from ..ingest import upload as ingest
from ..transcribe import whisper_engine
from ..adapt import providers, prompts, glossary, batch_manager as BM
from ..adapt import intro_outro as IO
from ..dubbing import package as PKG
from ..dubbing.router import get_router
from ..dubbing.vram_manager import check_model_fits
from ..dubbing.workers.protocol import TARGET_LANG
from ..directaudio.direct import synth_direct
from ..export import metadata as META
from ..export.subtitle_mask import MASK_TYPES, MASK_COLORS
from . import helpers as H


def _export_done_msg(pid: str, final_name: str) -> str:
    from ..common import stageflow as SF
    SF.mark_stage(pid, "export")
    return f"✅ Export complete: {final_name}"


# =====================================================================
# TAB 1 — Ingest & Transcribe
# =====================================================================
def build_tab1():
    cfg = C.load_config()
    with gr.Tab("1 · Ingest & Transcribe"):
        gr.Markdown("### Step 1 — Create a Chinese timestamped source transcript")
        with gr.Row():
            proj = gr.Textbox(label="Project name", placeholder="my_manga_ep1")
            src_lang = gr.Dropdown(
                ["Auto", "Mandarin", "Cantonese", "Other Chinese dialect"],
                value="Auto", label="Source language")
        with gr.Accordion("🗺️ Resume a project (progress + where to continue)", open=True):
            with gr.Row():
                resume_pick = gr.Dropdown(H.list_projects(), label="Load an existing project")
                resume_refresh = gr.Button("↻ Refresh list")
                resume_load = gr.Button("📂 Load & show progress", variant="primary")
            wf_status = gr.Markdown("_Pick a project and click Load, or start a new one below._")

            def _resume_refresh():
                return gr.update(choices=H.list_projects())

            def _resume_load(name):
                from ..common import stageflow as SF
                if not name:
                    return "", SF.resume_dashboard("")
                pid = safe_name(name)
                # load into the project name field so every tab is now on this project
                return name, SF.resume_dashboard(pid)

            resume_refresh.click(_resume_refresh, outputs=resume_pick)
            resume_load.click(_resume_load, resume_pick, [proj, wf_status])
            resume_pick.change(_resume_load, resume_pick, [proj, wf_status])
        with gr.Row():
            warm_btn = gr.Button("🔥 Download & load Whisper on GPU now "
                                 "(warm up while you upload)", variant="secondary")
            warm_status = gr.Markdown("_Whisper not warmed yet — first transcribe "
                                      "will load the model._")
        with gr.Accordion("① Add source video (upload / folder / Drive)", open=True):
            upload = gr.File(label="Browser upload video (best for small/medium files)",
                             file_types=["video"])
            gr.Markdown(
                "**For multi-GB video files:** use the Lightning Studio **Files** panel to "
                "upload directly into `data/input/`, then enter a project name. Auto-ingest "
                "copies the completed file safely into that project. Browser uploads can be "
                "interrupted by the Studio/reverse-proxy connection on very large files.")
            with gr.Row():
                folder_pick = gr.Dropdown(ingest.list_input_videos(),
                                          label="…or pick from data/input/")
                refresh = gr.Button("↻ Refresh input list")
                check_input_btn = gr.Button("✓ Check input readiness")
            auto_input = gr.Checkbox(
                True,
                label="Auto-ingest a completed video copied into data/input/",
                info="Set the project name first. The app waits for the file to stop changing; it does not auto-transcribe.")
            drive = gr.Textbox(label="…or Google Drive link (gdown)")
            ingest_btn = gr.Button("Ingest video", variant="primary")
        with gr.Accordion("② Transcribe", open=True):
            _chunk_default = int(cfg.get("whisper", {}).get("max_speech_s", 30))
            gr.Markdown("_Control how much audio Whisper transcribes per cue. "
                        "Bigger = fewer, longer cues (≈ 60 ÷ seconds cues per minute). "
                        "e.g. 30s ≈ 2 cues/min, 15s ≈ 4 cues/min._")
            chunk_len = gr.Slider(5, 45, value=_chunk_default, step=1,
                                  label="Whisper chunk length (seconds per cue)")
            transcribe_btn = gr.Button("Transcribe (large-v3)", variant="primary")
            show_ts = gr.Checkbox(True, label="Show timestamps")
        vpreview = gr.Video(label="Source video preview")
        status = gr.Markdown()
        transcript_box = gr.Textbox(label="Chinese transcript", lines=12)

        def do_refresh():
            return gr.update(choices=ingest.list_input_videos())

        def do_check_input(folder):
            if not folder:
                return "Pick a video from data/input/ first."
            r = ingest.check_input_video_ready(folder)
            return ("✅ " if r.get("ok") else "⏳ ") + r.get("message", "Unknown input status.")

        def _render_transcript(pid, with_ts):
            import json as _json
            tj = project_dir(pid) / "transcript" / "transcript.json"
            if not tj.exists():
                return ""
            segs = _json.loads(tj.read_text(encoding="utf-8"))
            # Prepend a metadata header so the user can SEE how transcription ran
            # (device, detected language, duration, resolution, cue count). If it
            # ran on CPU, this makes the slowness cause visible — not hidden.
            header = ""
            mj = project_dir(pid) / "transcript" / "transcript_meta.json"
            if mj.exists():
                try:
                    m = _json.loads(mj.read_text(encoding="utf-8"))
                    dev = m.get("device", "?")
                    warn = "  ⚠ CPU = SLOW (see Live Logs to enable GPU)" if dev == "cpu" else ""
                    lp = m.get("language_probability")
                    lp_s = f" ({lp:.0%})" if isinstance(lp, (int, float)) else ""
                    header = (
                        f"— {len(segs)} cues • lang={m.get('detected_language','?')}{lp_s}"
                        f" • {m.get('duration_s','?')}s"
                        f" • {m.get('width','?')}x{m.get('height','?')}"
                        f" • device={dev}/{m.get('compute_type','?')}{warn} —\n\n")
                except Exception:
                    pass
            if with_ts:
                def fmt(t):
                    m, s = divmod(int(t), 60)
                    return f"{m:02d}:{s:02d}.{int((t%1)*1000):03d}"
                return header + "\n".join(
                    f"[{fmt(s['start'])} → {fmt(s['end'])}] {s['text']}" for s in segs)
            return header + "\n".join(s["text"] for s in segs)

        def do_ingest(name, up, folder, drive_url):
            if not name:
                return None, "Enter a project name first.", ""
            pid = H.create_project(name)
            path = None
            if up is not None:
                path = ingest.store_uploaded(up.name if hasattr(up, "name") else up,
                                             pid, Path(getattr(up, "name", up)).name)
            elif folder:
                dst = project_dir(pid) / "source" / Path(folder).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                import shutil; shutil.copy(folder, dst); path = str(dst)
            elif drive_url:
                r = ingest.download_drive(drive_url, pid)
                if not r.get("ok"):
                    return None, f"Drive error: {r.get('error')}", ""
                path = r["path"]
            else:
                return None, "Provide an upload, folder pick, or Drive link.", ""
            return path, f"Ingested into project **{pid}**.", ""

        def do_auto_ingest(name, enabled):
            """Timer callback: safely pick up a fully copied data/input video."""
            choices = ingest.list_input_videos()
            if not enabled or not (name or "").strip():
                return gr.update(choices=choices), gr.skip(), gr.skip()
            pid = H.create_project(name)
            r = ingest.auto_ingest_stable_input(pid)
            if r.get("ok") and not r.get("already"):
                return (gr.update(choices=choices, value=None), r["path"],
                        f"✅ {r['message']}  Ready to transcribe when you choose.")
            return gr.update(choices=choices), gr.skip(), gr.skip()

        def do_warm(progress=gr.Progress()):
            progress(0.1, desc="Downloading + loading Whisper onto the GPU…")
            r = whisper_engine.warm_start(progress=lambda m: progress(0.5, desc=str(m)))
            if not r.get("ok"):
                return (f"⚠ Whisper warm-up did not confirm GPU: {r.get('error', '')} "
                        f"(transcribe still works; check Live Logs).")
            dev = r.get("device", "?")
            if dev == "cpu":
                return ("⚠ Whisper loaded on **CPU** (GPU unavailable) — see Live Logs "
                        "for the reason. Transcription will be slow.")
            return (f"✅ Whisper is **warm on {dev}** ({r.get('compute','')}). "
                    f"Transcribe will start instantly. It auto-frees VRAM before dubbing.")

        def do_transcribe(name, lang, with_ts, chunk_s, progress=gr.Progress()):
            pid = safe_name(name)
            src_v = find_source_video(pid)
            if not src_v:
                return "No source video found — ingest first."
            vids = [src_v]
            out = project_dir(pid) / "transcript"
            progress(0.1, desc="Transcribing… (warm Whisper = instant; else loads model)")
            r = whisper_engine.transcribe(
                str(vids[0]), str(out), lang, chunk_seconds=int(chunk_s),
                progress=lambda m: progress(0.5, desc=str(m)))
            if not r.get("ok"):
                return f"Transcription failed: {r.get('error')}"
            progress(0.95, desc="Rendering transcript…")
            from ..common import stageflow as SF
            SF.mark_stage(pid, "ingest"); SF.mark_stage(pid, "transcribe")
            return _render_transcript(pid, with_ts)

        def do_toggle_ts(name, with_ts):
            return _render_transcript(safe_name(name), with_ts) if name else ""

        refresh.click(do_refresh, outputs=folder_pick)
        check_input_btn.click(do_check_input, folder_pick, status)
        # Poll only the lightweight folder listing. A completed file is copied
        # atomically into the selected project's source folder; transcription stays
        # an explicit user action because it is expensive.
        input_timer = gr.Timer(3.0)
        input_timer.tick(do_auto_ingest, [proj, auto_input], [folder_pick, vpreview, status])
        ingest_btn.click(do_ingest, [proj, upload, folder_pick, drive],
                         [vpreview, status, transcript_box])
        warm_btn.click(do_warm, outputs=warm_status)
        transcribe_btn.click(do_transcribe, [proj, src_lang, show_ts, chunk_len],
                             transcript_box)
        show_ts.change(do_toggle_ts, [proj, show_ts], transcript_box)
    return {"project": proj}


# =====================================================================
# TAB 2 — Script & Adaptation
# =====================================================================
def build_tab2():
    with gr.Tab("2 · Script & Adaptation"):
        gr.Markdown("### Build the target narration script, then forward it to Dubbing")
        with gr.Row():
            proj = gr.Dropdown(H.list_projects(), label="Project")
            target = gr.Dropdown(H.target_choices(), label="Target",
                                 value="hinglish_devanagari")
            dub_model = gr.Dropdown(H.model_choices(), value="indicf5",
                                    label="Dubbing model (forwarded to Tab 3)")
            refresh = gr.Button("↻")
        with gr.Row():
            prewarm_btn = gr.Button("⬇ Download & warm up this model now (optional)")
            prewarm_status = gr.Markdown()
        gr.Markdown("_Tip: press the button above while you review/adapt the script — "
                    "the model (~4-5 GB) downloads in the background so dubbing starts "
                    "instantly later. Uses no GPU during adaptation._")
        with gr.Tabs():
            with gr.Tab("A · Manual"):
                manual = gr.Textbox(label="Paste full adaptation (one line per cue)", lines=8)
                save_manual = gr.Button("Save manual as adaptation")
            with gr.Tab("B · Import SRT"):
                srt_file = gr.File(label="Target SRT (match source cue count)",
                                   file_types=[".srt"])
                import_srt = gr.Button("Import SRT as adaptation")
            with gr.Tab("C/D · AI adaptation"):
                with gr.Row():
                    provider = gr.Dropdown(providers.PROVIDERS, value="gemini",
                                           label="Provider")
                    model = gr.Dropdown(
                        providers.model_choices("gemini"),
                        value=providers.default_model("gemini"),
                        label="AI model (pick one — no ID typing)",
                        allow_custom_value=True)
                    refresh_model_dd = gr.Button("↻ Models")
                with gr.Accordion("Model browser (live)", open=False):
                    with gr.Row():
                        f_text = gr.Checkbox(True, label="Text-capable only")
                        f_free = gr.Checkbox(False, label="Free / free-plan")
                        f_json = gr.Checkbox(False, label="Structured JSON")
                    search = gr.Textbox(label="Search model name")
                    refresh_models = gr.Button("Refresh model list")
                    model_table = gr.Dataframe(
                        headers=["id", "provider", "text", "free", "json", "context", "notes"],
                        label="Available models")
                with gr.Row():
                    style = gr.Dropdown(H.style_choices(),
                                        value="Engaging YouTube Hinglish", label="Style")
                    retention = gr.Dropdown(prompts.retention_choices(),
                                            value="Full Duration Dub",
                                            label="🎬 Adaptation mode (duration & pacing)")
                    engagement = gr.Dropdown(prompts.engagement_choices(),
                                             value="Natural Commentary",
                                             label="🎙 Audience engagement")
                    auto_gloss = gr.Checkbox(True, label="Auto glossary & name consistency")
                with gr.Accordion("💾 Full-setup presets (one-click per new video)", open=False):
                    gr.Markdown("_Save your whole setup (target + style + retention + "
                                "emotions) as a named preset, then load it for the next video._")
                    with gr.Row():
                        setup_name = gr.Textbox(label="Preset name")
                        save_setup = gr.Button("Save current setup")
                        setup_pick = gr.Dropdown(prompts.list_setup_presets(),
                                                 label="Load preset")
                        load_setup = gr.Button("Load preset")
                        del_setup = gr.Button("Delete preset")
                    setup_status = gr.Markdown()
                with gr.Accordion("🎭 Emotions (model-aware, ON by default)", open=False):
                    add_emotions = gr.Checkbox(
                        True, label="Add emotions for selected TTS model "
                                    "(auto-applies only on models that support it)")
                    ai_free_emo = gr.Checkbox(
                        False, label="AI free mode (off = curated manga palette)")
                    emo_note = gr.Markdown(
                        "_On by default. It only affects Fish & VoxCPM2 (which read "
                        "emotion from text); other models simply ignore it — no harm._")
                project_prompt = gr.Textbox(label="Project-Specific Prompt Addition", lines=2)
                with gr.Row():
                    tmpl_name = gr.Textbox(label="Template name")
                    save_tmpl = gr.Button("Save as template")
                    tmpl_pick = gr.Dropdown(prompts.list_templates(), label="Load template")
                    load_tmpl = gr.Button("Load template")
                    del_tmpl = gr.Button("Delete template")
                show_prompt = gr.Button("Show Effective Prompt Sent to AI")
                eff_prompt = gr.Textbox(label="Exact adaptation prompt + cue payload preview", lines=18)

        gr.Markdown("#### Translation Batch Manager")
        with gr.Row():
            main_batches = gr.Slider(1, 12, value=6, step=1, label="Main batches")
            create_plan = gr.Button("Create / Reset Plan")
            translate_all = gr.Button("Translate All Pending", variant="primary")
        batch_table = gr.Dataframe(
            headers=["batch", "cue_count", "status", "active_version",
                     "provider", "model", "context_status", "error"],
            label="Batches")
        with gr.Row():
            sel_batch = gr.Number(1, label="Selected batch", precision=0)
            view_batch = gr.Button("View Selected Batch")
            retry_batch = gr.Button("Retry Selected Batch")
        with gr.Row():
            active_ver = gr.Dropdown([], label="Active version (pick which is live)")
            list_vers = gr.Button("List versions")
            set_active = gr.Button("Set as active version")
        batch_view = gr.Textbox(label="Batch content", lines=6)

        gr.Markdown("#### Assembled Adaptation (edit if needed, then forward)")
        assembled = gr.Textbox(label="Full target narration (one line per cue)", lines=10)
        with gr.Row():
            load_assembled = gr.Button("Load / Refresh Assembled Adaptation")
            duration_dashboard_btn = gr.Button("⏱ Check duration fit")
            repair_short_btn = gr.Button("🛠 Repair short cues with AI")
            forward_btn = gr.Button("💾 Save + Forward to Dubbing", variant="primary")
        with gr.Accordion("🔎 Back-translation quality check (optional)", open=False):
            gr.Markdown(
                "A second AI pass compares each adapted line against the original "
                "Chinese cue and flags any that lost or changed meaning. Uses the "
                "provider/model selected above.")
            backcheck_btn = gr.Button("Run Back-Translation Check")
            backcheck_out = gr.Textbox(label="Quality report", lines=8)
        status = gr.Markdown()

        def do_refresh():
            return (gr.update(choices=H.list_projects()),
                    gr.update(choices=prompts.list_templates()))

        def do_effective(p, t, s, pp, retn, eng, dubmid, emo_on, emo_free):
            """Show the actual system prompt plus a representative user payload."""
            from ..adapt import emotions as EMO
            from ..adapt import quality as Q
            cues = []
            if p:
                tr = project_dir(safe_name(p)) / "transcript" / "transcript.json"
                if tr.exists():
                    cues = json.loads(tr.read_text(encoding="utf-8"))
            g = glossary.load(safe_name(p), t) if p else None
            sysp = prompts.build_effective_prompt(t, s, pp, glossary=g, retention=retn, engagement=eng)
            sysp += "\n" + Q.duration_rules(cues)
            sysp += "\n" + Q.CUE_JSON_INSTRUCTIONS
            if g:
                lock = Q.glossary_lock_block(g)
                if lock:
                    sysp += "\n" + lock
                sysp += "\n" + Q.GLOSSARY_INSTRUCTIONS
            if emo_on and EMO.is_emotion_capable(dubmid):
                sysp += EMO.build_emotion_prompt(dubmid, emo_free)
            preview = Q.build_cue_payload(cues[:5]) if cues else "(Select a transcribed project to preview cue payload.)"
            return "SYSTEM PROMPT SENT TO AI:\n\n" + sysp + "\n\n---\nUSER CUE PAYLOAD PREVIEW:\n\n" + preview

        def do_refresh_models(prov, ft, ff, fj, q):
            rows = providers.list_models(prov)
            def keep(r):
                if ft and not r["text"]:
                    return False
                if ff and not r["free"]:
                    return False
                if fj and not r["json"]:
                    return False
                if q and q.lower() not in r["id"].lower():
                    return False
                return True
            rows = [r for r in rows if keep(r)]
            return [[r["id"], r["provider"], r["text"], r["free"], r["json"],
                     r["context"], r["notes"]] for r in rows]

        def _model_dd_for(prov):
            """Refresh the model dropdown choices for the chosen provider (live +
            curated fallback), selecting that provider's default."""
            choices = providers.model_choices(prov)
            dm = providers.default_model(prov)
            val = dm if dm in choices else (choices[0] if choices else None)
            return gr.update(choices=choices, value=val)

        def _rows(pid, t):
            plan = BM.load_plan(pid, t)
            if not plan:
                return []
            return [[b["batch"], b["cue_count"], b["status"], b["active_version"],
                     b["provider"], b["model"], b["context_status"], b["error"]]
                    for b in plan["batches"]]

        def do_create_plan(p, t, n):
            if not p:
                return None, "Pick a project."
            tr = project_dir(safe_name(p)) / "transcript" / "transcript.json"
            if not tr.exists():
                return None, "Transcribe first (Tab 1)."
            cues = json.loads(tr.read_text(encoding="utf-8"))
            plan = BM.create_plan(safe_name(p), t, cues, int(n))
            return _rows(safe_name(p), t), f"Plan created: {len(plan['batches'])} batches."

        def _translate_batch(pid, t, b, prov, mdl, st, pp, ag, emo_on, emo_free, dubmid,
                             retn="", eng="Natural Commentary"): 
            from ..adapt import emotions as EMO
            from ..adapt import quality as Q
            tr = json.loads((project_dir(pid) / "transcript" / "transcript.json")
                            .read_text(encoding="utf-8"))
            g = glossary.load(pid, t) if ag else None
            cues = tr[b["cue_lo"]:b["cue_hi"]]
            n_cues = len(cues)

            # --- Cross-batch context carryover (upgrade #3) ---
            prior_tail = ""
            if b["cue_lo"] > 0:
                prev_lines = BM.assemble_adaptation(pid, t)
                prior_tail = Q.summarise_tail(prev_lines)

            # --- Base layered prompt + glossary lock (upgrade #4 reuse) ---
            sysp = prompts.build_effective_prompt(t, st, pp, glossary=g, retention=retn, engagement=eng)
            sysp += "\n" + Q.duration_rules(cues)          # upgrade #2
            sysp += "\n" + Q.CUE_JSON_INSTRUCTIONS         # upgrade #1
            if ag:
                lock = Q.glossary_lock_block(g)
                if lock:
                    sysp += "\n" + lock
                sysp += "\n" + Q.GLOSSARY_INSTRUCTIONS     # upgrade #4
            ctx = Q.build_context_block(prior_tail)         # upgrade #3
            if ctx:
                sysp += "\n" + ctx
            # model-aware emotion layer (only if enabled AND model supports text emotion)
            if emo_on and EMO.is_emotion_capable(dubmid):
                sysp += EMO.build_emotion_prompt(dubmid, emo_free)

            # Duration-aware, numbered JSON payload (upgrades #1 + #2)
            user_content = Q.build_cue_payload(cues)
            r = providers.adapt(prov, mdl, sysp, user_content, want_json=True)
            used = mdl or providers.default_model(prov)
            if not r.get("ok"):
                BM.mark(pid, t, b["batch"], status="Paused", error=r.get("error", ""),
                        provider=prov, model=used)
                return False

            # Strict per-cue alignment with graceful fallback (upgrade #1)
            lines, warns = Q.parse_cue_response(r["text"], n_cues)
            # Auto-merge glossary the AI returned (upgrade #4)
            if ag:
                new_g = Q.extract_glossary_from_response(r["text"])
                if new_g:
                    glossary.merge(pid, t, new_g)
            ver = BM.save_batch_version(pid, t, b["batch"], lines, prov, used)
            ctx_status = "OK"
            err = ""
            if warns:
                ctx_status = "Check alignment"
                err = " | ".join(warns)[:300]
            BM.mark(pid, t, b["batch"], status="Done", active_version=ver,
                    provider=prov, model=used, error=err, context_status=ctx_status)
            return True

        def do_translate_all(p, t, prov, mdl, st, pp, ag, emo_on, emo_free, dubmid, retn, eng):
            if not p:
                return None, "Pick a project."
            pid = safe_name(p)
            plan = BM.load_plan(pid, t)
            if not plan:
                return None, "Create a plan first."
            from ..adapt import emotions as EMO
            note = ""
            if emo_on and not EMO.is_emotion_capable(dubmid):
                note = f" (Emotions skipped — {EMO.capability_note(dubmid)})"
            for b in plan["batches"]:
                if b["status"] == "Done":
                    continue
                if not _translate_batch(pid, t, b, prov, mdl, st, pp, ag,
                                        emo_on, emo_free, dubmid, retn, eng):
                    break
            return _rows(pid, t), f"Translation pass complete.{note}"

        def do_view_batch(p, t, bn):
            if not p:
                return "Pick a project."
            return BM.get_batch_text(safe_name(p), t, int(bn)) or "(no content for this batch)"

        def do_retry_batch(p, t, bn, prov, mdl, st, pp, ag, emo_on, emo_free, dubmid, retn, eng):
            if not p:
                return None, "Pick a project.", ""
            pid = safe_name(p)
            plan = BM.load_plan(pid, t)
            if not plan:
                return None, "Create a plan first.", ""
            b = next((x for x in plan["batches"] if x["batch"] == int(bn)), None)
            if not b:
                return _rows(pid, t), "No such batch.", ""
            ok = _translate_batch(pid, t, b, prov, mdl, st, pp, ag, emo_on, emo_free, dubmid, retn, eng)
            BM.mark_later_needs_context(pid, t, int(bn))
            txt = BM.get_batch_text(pid, t, int(bn))
            return _rows(pid, t), ("Retried ✅" if ok else "Retry failed (see error col)"), txt

        def do_list_versions(p, t, bn):
            if not p:
                return gr.update(choices=[]), "Pick a project."
            vers = BM.list_batch_versions(safe_name(p), t, int(bn))
            if not vers:
                return gr.update(choices=[]), f"Batch {int(bn)} has no versions yet."
            return (gr.update(choices=[str(v) for v in vers], value=str(vers[-1])),
                    f"Batch {int(bn)} versions: {vers}")

        def do_set_active(p, t, bn, ver):
            if not p or not ver:
                return None, "Pick a project and a version.", ""
            pid = safe_name(p)
            BM.mark(pid, t, int(bn), active_version=int(ver))
            txt = BM.get_batch_text(pid, t, int(bn), int(ver))
            return _rows(pid, t), f"Batch {int(bn)} active version set to v{int(ver)}.", txt

        def do_load_assembled(p, t):
            if not p:
                return "", "Pick a project."
            lines = BM.assemble_adaptation(safe_name(p), t)
            if not lines:
                return "", "No completed batches yet — translate first (or paste manual)."
            return "\n".join(lines), f"Assembled {len(lines)} lines from active batch versions."

        def _cue_count(pid):
            tr = project_dir(pid) / "transcript" / "transcript.json"
            if not tr.exists():
                return None
            try:
                return len(json.loads(tr.read_text(encoding="utf-8")))
            except Exception:
                return None

        def _count_note(pid, n_lines):
            """L-2: warn when the adaptation line count != transcript cue count."""
            n_cues = _cue_count(pid)
            if n_cues is None:
                return " (no transcript found — transcribe first so I can verify cue count)."
            if n_lines == n_cues:
                return f" ✅ matches the {n_cues} source cues."
            return (f" ⚠️ WARNING: {n_lines} lines but {n_cues} source cues. "
                    f"They MUST match 1:1 or the dub will misalign / drop cues. "
                    f"Add or remove lines so the counts are equal before forwarding.")

        def do_save_manual(p, t, text):
            if not p:
                return "", "Pick a project."
            pid = safe_name(p)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            note = _count_note(pid, len(lines))
            return "\n".join(lines), f"Loaded {len(lines)} manual lines into Assembled box.{note}"

        def do_import_srt(p, t, srt):
            if not p or not srt:
                return "", "Pick a project and an SRT file."
            pid = safe_name(p)
            from ..export.srt import read_srt
            cues = read_srt(srt.name if hasattr(srt, "name") else srt)
            lines = [c.text.replace("\n", " ").strip() for c in cues]
            note = _count_note(pid, len(lines))
            return "\n".join(lines), f"Imported {len(lines)} lines from SRT into Assembled box.{note}"

        def do_backcheck(p, t, prov, mdl, text):
            from ..adapt import quality as Q
            if not p:
                return "Pick a project."
            pid = safe_name(p)
            adapted = [ln for ln in (text or "").splitlines() if ln.strip()]
            if not adapted:
                return "Load / assemble the adaptation first (box is empty)."
            tr_path = project_dir(pid) / "transcript" / "transcript.json"
            if not tr_path.exists():
                return "No transcript found — transcribe first (Tab 1)."
            cues = json.loads(tr_path.read_text(encoding="utf-8"))
            if len(adapted) != len(cues):
                note = (f"⚠️ Note: {len(adapted)} adapted lines vs {len(cues)} "
                        f"source cues — checking the overlapping range.\n")
            else:
                note = ""
            payload = Q.build_backcheck_payload(cues, adapted)
            r = providers.adapt(prov, mdl, Q.BACKCHECK_INSTRUCTIONS, payload,
                                want_json=True)
            if not r.get("ok"):
                return note + f"Back-check call failed: {r.get('error', 'unknown')}"
            checks = Q.parse_backcheck(r["text"])
            return note + Q.backcheck_summary(checks)

        def _duration_wps(t, mid):
            return 3.0 if mid == "voxcpm2" and t in ("hinglish_roman", "hinglish_devanagari") else 2.6

        def do_duration_dashboard(p, t, mid, text):
            from ..adapt import quality as Q
            if not p or not text:
                return "Pick a project and load an adaptation first."
            trp = project_dir(safe_name(p)) / "transcript" / "transcript.json"
            if not trp.exists():
                return "Transcribe first."
            cues = json.loads(trp.read_text(encoding="utf-8"))
            lines = [x for x in text.splitlines() if x.strip()]
            fit = Q.duration_fit(cues, lines, _duration_wps(t, mid))
            short = [r for r in fit["rows"] if r["short"]]
            duplicate_pairs = [i + 1 for i in range(1, len(cues))
                               if (cues[i].get("text") or "").strip()
                               and (cues[i].get("text") or "").strip() == (cues[i-1].get("text") or "").strip()]
            duplicate_note = (f"\n⚠ Adjacent duplicate transcript cues: {duplicate_pairs[:20]}. "
                              "Review them before repair; they are not deleted automatically."
                              if duplicate_pairs else "")
            return (f"**Source cue duration:** {fit['source_seconds']/60:.1f} min  ·  "
                    f"**Predicted narration:** {fit['predicted_seconds']/60:.1f} min  ·  "
                    f"**Under-length cues:** {len(short)}/{len(cues)}  ·  "
                    f"**Words:** {fit['words']}\n\n"
                    + (f"First short cues: {[r['idx']+1 for r in short[:20]]}" if short else "✅ All cues meet the selected duration target.")
                    + duplicate_note)

        def do_repair_short(p, t, mid, text, prov, mdl, progress=gr.Progress()):
            from ..adapt import quality as Q
            if not p or not text:
                return text, "Pick a project and load an adaptation first."
            trp = project_dir(safe_name(p)) / "transcript" / "transcript.json"
            if not trp.exists():
                return text, "Transcribe first."
            cues = json.loads(trp.read_text(encoding="utf-8"))
            lines = [x for x in text.splitlines() if x.strip()]
            if len(lines) != len(cues):
                return text, "Line count must match source cue count before repair."
            fit = Q.duration_fit(cues, lines, _duration_wps(t, mid))
            short = [r for r in fit["rows"] if r["short"]]
            if not short:
                return text, "✅ No under-length cues need repair."
            chosen = [cues[r["idx"]] for r in short]
            prompt = ("You repair UNDER-LENGTH narration cues for a duration-faithful manga dub. "
                      "Return exactly one JSON cue per input cue, same order. Expand each cue "
                      "to its supplied seconds using only source-supported action, scene context, "
                      "reaction, cause and consequence. Do not invent plot, merge cues, or summarize. "
                      "Return spoken narration only.\n" + Q.CUE_JSON_INSTRUCTIONS + "\n" + Q.duration_rules(chosen))
            progress(0.2, desc=f"Repairing {len(short)} short cues…")
            r = providers.adapt(prov, mdl, prompt, Q.build_cue_payload(chosen), want_json=True)
            if not r.get("ok"):
                return text, f"Repair failed: {r.get('error', 'unknown')}"
            repaired, warns = Q.parse_cue_response(r["text"], len(chosen))
            for row, replacement in zip(short, repaired):
                if replacement.strip():
                    lines[row["idx"]] = replacement.strip()
            progress(1.0, desc="Duration repair complete")
            return "\n".join(lines), f"✅ Repaired {len(short)} under-length cues." + (" Warnings: " + " | ".join(warns) if warns else "")

        def do_forward(p, t, mid, text):
            if not p:
                return "Pick a project."
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return "Assembled adaptation is empty — load/translate/paste first."
            # L-2 gate: block forwarding a mismatched line/cue count (would drop/misalign cues)
            n_cues = _cue_count(safe_name(p))
            if n_cues is not None and len(lines) != n_cues:
                return (f"⛔ Not forwarded: {len(lines)} narration lines but {n_cues} "
                        f"source cues. They must match 1:1 (one line per cue) or the "
                        f"dub will misalign and cues can be dropped. Fix the line count "
                        f"in the Assembled box, then forward again.")
            # Duration analysis is advisory: users may intentionally create a short
            # recap, so never block forwarding solely because it is under-length.
            import re
            duration_warn = ""
            tr_path = project_dir(safe_name(p)) / "transcript" / "transcript.json"
            if tr_path.exists():
                cues = json.loads(tr_path.read_text(encoding="utf-8"))
                source_s = sum(max(0.0, float(c.get("end", 0)) - float(c.get("start", 0)))
                               for c in cues)
                spoken = [re.sub(r"^\s*\([^)]{0,100}\)\s*", "", ln) for ln in lines]
                words = sum(len(re.findall(r"\b[\w'-]+\b", ln, flags=re.UNICODE))
                            for ln in spoken)
                # VoxCPM2 delivers Roman Hinglish around 3 words/sec in real channel
                # narration.  Report a useful duration estimate, but do not force a
                # creator who deliberately wants a recap to regenerate it.
                target_wps = 3.0 if (mid == "voxcpm2" and t in ("hinglish_roman", "hinglish_devanagari")) else 2.6
                estimated_s = words / target_wps if words else 0.0
                if source_s >= 60 and estimated_s < source_s * 0.85:
                    needed = int(source_s * 0.85 * target_wps)
                    duration_warn = (f"\n⚠️ **Duration warning:** estimated narration is "
                                     f"≈{estimated_s / 60:.1f} min for a "
                                     f"{source_s / 60:.1f} min source ({words} words; "
                                     f"≈{needed}+ words for an 85% match). "
                                     "This will create a shorter cue-locked video unless "
                                     "you intentionally want a recap.")
            ver = PKG.forward_package(safe_name(p), t, {
                "target": t, "narration_lines": lines,
                "dubbing_model": mid, "source": "tab2"})
            from ..common import stageflow as SF
            warn = ""
            if SF.transcript_changed_since_adaptation(safe_name(p)):
                warn = ("\n⚠️ Note: the transcript changed since your last "
                        "adaptation — re-check that lines still match cues.")
            SF.mark_stage(safe_name(p), "adaptation"); SF.mark_stage(safe_name(p), "forward")
            SF.record_transcript_fingerprint(safe_name(p))  # L-3: lock the contract
            return (f"✅ Forwarded {len(lines)} lines to Dubbing as package {ver} "
                    f"(model: {C.model_cfg(mid)['label']}).{warn}{duration_warn}")

        refresh.click(do_refresh, outputs=[proj, tmpl_pick])
        show_prompt.click(do_effective,
                          [proj, target, style, project_prompt, retention, engagement, dub_model,
                           add_emotions, ai_free_emo], eff_prompt)
        refresh_models.click(do_refresh_models,
                             [provider, f_text, f_free, f_json, search], model_table)
        # Auto-fill the model dropdown when the provider changes, and on ↻ Models.
        provider.change(_model_dd_for, provider, model)
        refresh_model_dd.click(_model_dd_for, provider, model)
        create_plan.click(do_create_plan, [proj, target, main_batches],
                          [batch_table, status])
        def do_emo_note(dubmid, emo_on):
            from ..adapt import emotions as EMO
            cap = EMO.is_emotion_capable(dubmid)
            note = EMO.capability_note(dubmid)
            if cap and emo_on:
                note += "\n\n" + EMO.palette_reference(dubmid)
            # grey out the checkboxes if the model can't use text emotion
            return (note,
                    gr.update(interactive=cap, value=(emo_on and cap)),
                    gr.update(interactive=cap))

        translate_all.click(do_translate_all,
                            [proj, target, provider, model, style, project_prompt,
                             auto_gloss, add_emotions, ai_free_emo, dub_model, retention, engagement],
                            [batch_table, status])
        view_batch.click(do_view_batch, [proj, target, sel_batch], batch_view)
        retry_batch.click(do_retry_batch,
                          [proj, target, sel_batch, provider, model, style,
                           project_prompt, auto_gloss, add_emotions, ai_free_emo,
                           dub_model, retention, engagement],
                          [batch_table, status, batch_view])
        dub_model.change(do_emo_note, [dub_model, add_emotions],
                         [emo_note, add_emotions, ai_free_emo])
        add_emotions.change(do_emo_note, [dub_model, add_emotions],
                            [emo_note, add_emotions, ai_free_emo])
        list_vers.click(do_list_versions, [proj, target, sel_batch], [active_ver, status])
        set_active.click(do_set_active, [proj, target, sel_batch, active_ver],
                         [batch_table, status, batch_view])
        load_assembled.click(do_load_assembled, [proj, target], [assembled, status])
        duration_dashboard_btn.click(do_duration_dashboard, [proj, target, dub_model, assembled], status)
        repair_short_btn.click(do_repair_short, [proj, target, dub_model, assembled, provider, model], [assembled, status])
        save_manual.click(do_save_manual, [proj, target, manual], [assembled, status])
        import_srt.click(do_import_srt, [proj, target, srt_file], [assembled, status])
        def do_prewarm(mid, progress=gr.Progress()):
            """Download + install the dubbing model NOW (during adaptation) so the
            first dub starts instantly. Streams install progress; no GPU used here."""
            from ..dubbing.installer import is_installed, install_model
            from ..common.diskmanager import fits_budget
            if not mid:
                yield "Pick a dubbing model first."; return
            # Free the resident Whisper GPU worker NOW — the user is bringing a TTS
            # model in, and both can't share the 16 GB T4. (Also auto-fires on load.)
            try:
                from ..transcribe import whisper_engine as _we
                if _we.warm_status().get("alive"):
                    _we.release_gpu(reason="user is loading a TTS model (prewarm)")
                    yield "🔻 Freed Whisper from GPU to make room for the TTS model…"
            except Exception:
                pass
            label = C.model_cfg(mid).get("label", mid)
            if is_installed(mid):
                yield f"✅ {label} is already installed — dubbing will start instantly."
                return
            ok, msg = fits_budget(mid)
            if not ok:
                yield f"❌ {msg}"; return
            progress(0.05, desc=f"Downloading {label}…")
            yield f"⬇ Installing {label} (~several minutes; downloads weights)…"
            state = {"last": ""}
            def _p(m):
                state["last"] = str(m)
            r = install_model(mid, progress=_p)
            if r.get("ok"):
                yield f"✅ {label} installed & ready — dubbing will start instantly now."
            else:
                yield (f"❌ Install failed: {r.get('message','?')}\n\n"
                       f"See the Live Logs tab for details.")

        prewarm_btn.click(do_prewarm, dub_model, prewarm_status, concurrency_limit=None)
        forward_btn.click(do_forward, [proj, target, dub_model, assembled], status)
        backcheck_btn.click(do_backcheck,
                            [proj, target, provider, model, assembled], backcheck_out)
        def do_save_tmpl(n, pp):
            msg = prompts.save_template(n, pp)
            return msg, gr.update(choices=prompts.list_templates())

        def do_load_tmpl(n):
            txt = prompts.load_template(n)
            if not txt:
                return gr.update(), "No such template (or it is empty)."
            return gr.update(value=txt), f"Loaded template '{n}' into Project Prompt."

        def do_del_tmpl(n):
            msg = prompts.delete_template(n)
            return msg, gr.update(choices=prompts.list_templates())

        save_tmpl.click(do_save_tmpl, [tmpl_name, project_prompt], [status, tmpl_pick])
        load_tmpl.click(do_load_tmpl, tmpl_pick, [project_prompt, status])
        del_tmpl.click(do_del_tmpl, tmpl_pick, [status, tmpl_pick])

        # ---- Full-setup presets (#6) ----
        def do_save_setup(nm, tgt, sty, retn, emo_on, emo_free, pp):
            msg = prompts.save_setup_preset(nm, {
                "target": tgt, "style": sty, "retention": retn,
                "add_emotions": bool(emo_on), "ai_free_emo": bool(emo_free),
                "project_prompt": pp})
            return msg, gr.update(choices=prompts.list_setup_presets())

        def do_load_setup(nm):
            s = prompts.load_setup_preset(nm)
            if not s:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), "No such preset.")
            return (gr.update(value=s.get("target")),
                    gr.update(value=s.get("style")),
                    gr.update(value=s.get("retention", "None (use style only)")),
                    gr.update(value=s.get("add_emotions", False)),
                    gr.update(value=s.get("ai_free_emo", False)),
                    gr.update(value=s.get("project_prompt", "")),
                    f"Loaded preset '{nm}'.")

        def do_del_setup(nm):
            msg = prompts.delete_setup_preset(nm)
            return msg, gr.update(choices=prompts.list_setup_presets())

        save_setup.click(do_save_setup,
                         [setup_name, target, style, retention, add_emotions,
                          ai_free_emo, project_prompt],
                         [setup_status, setup_pick])
        load_setup.click(do_load_setup, setup_pick,
                         [target, style, retention, add_emotions, ai_free_emo,
                          project_prompt, setup_status])
        del_setup.click(do_del_setup, setup_pick, [setup_status, setup_pick])
    return {}


# =====================================================================
# TAB 3A — Professional Voice Design Studio
# =====================================================================
def build_voice_design_tab():
    from ..directaudio import voices as VOX
    with gr.Tab("3A · Voice Design Studio"):
        gr.Markdown("### Design, audition and approve one permanent manga narrator")
        with gr.Row():
            target = gr.Dropdown(H.target_choices(), value="hinglish_devanagari", label="Narrator language")
            count = gr.Slider(1, 6, value=4, step=1, label="Candidate voices")
        
        with gr.Accordion("🎙 Narrator Persona", open=True):
            persona = gr.Textbox(label="Voice design prompt", lines=3,
                value="A warm early-twenties Indian male narrator, medium-low clear voice, neutral North-Indian Hinglish accent, youthful but mature, confident manga storyteller, controlled cinematic energy, calm baseline, clear word endings, natural pace, expressive only at major twists")
            seed = gr.Textbox(label="Seed script and exact transcript", lines=3,
                value="नमस्ते दोस्तों, आज की कहानी में एक ऐसा twist आने वाला है जो hero की पूरी दुनिया बदल देगा। लेकिन असली सवाल यह है कि उसका next move उसे जीत दिलाएगा, या उसे एक नई मुसीबत में डाल देगा। चलो शुरू करते हैं।")
        
        with gr.Accordion("💾 Narrator Profile (save/load complete designs)", open=False):
            with gr.Row():
                profile_name = gr.Textbox(label="Profile name", placeholder="my_narrator_v1")
                save_profile = gr.Button("💾 Save current persona as profile")
                load_profile = gr.Button("📂 Load profile")
                del_profile = gr.Button("🗑 Delete profile")
            profile_pick = gr.Dropdown(VOX.list_narrator_profiles(), label="Saved profiles")
            profile_status = gr.Markdown()
        
        with gr.Accordion("🔀 Clone Mode (controls narrator identity in long dubs)", open=False):
            gr.Markdown("**Hi-Fi Clone** = prompt_wav + prompt_text + reference_wav → highest similarity, style prefix ignored.\n"
                        "**Controllable Clone** = reference_wav + style prefix → emotion/style controllable.\n"
                        "**Hybrid Smart Routing** = first cue Hi-Fi (if reference text provided), then Controllable for rest.")
            clone_mode = gr.Radio(
                ["Hybrid Smart Routing (recommended)", "Hi-Fi Clone (max fidelity)", "Controllable Clone (style control)"],
                value="Hybrid Smart Routing (recommended)",
                label="Clone mode for VoxCPM2"
            )
            gr.Markdown("_Hi-Fi needs reference transcript for exact cloning. Controllable uses your style prefix. "
                        "Hybrid uses Hi-Fi for cue 1 if transcript is given, then Controllable for consistency._")
        
        with gr.Row():
            generate = gr.Button("🎲 Generate designed voice candidates", variant="primary")
            promote = gr.Button("⭐ Set selected candidate as global default narrator", variant="primary")
        
        candidate = gr.Dropdown([], label="Audition candidate")
        player = gr.Audio(label="Candidate preview", interactive=False)
        
        with gr.Accordion("⭐ Candidate Scoring & Consistency Test", open=False):
            gr.Markdown("Rate candidates 0–100, then run the 5-line consistency suite to verify narrator stability across diverse lines.")
            with gr.Row():
                cand_score = gr.Slider(0, 100, value=80, step=1, label="Score (0–100)")
                cand_notes = gr.Textbox(label="Notes", placeholder="e.g. natural pace, clear endings, slight buzz at end")
                rate_btn = gr.Button("💾 Save rating")
            with gr.Row():
                consistency_btn = gr.Button("🧪 Run 5-line consistency test", variant="secondary")
            consistency_status = gr.Markdown()
            consistency_player = gr.Audio(label="Consistency test samples", interactive=False)
        
        status = gr.Markdown()
        
        with gr.Accordion("🔎 Reference & narrator checks", open=False):
            saved = gr.Dropdown(VOX.list_voices(), label="Saved personal/reference voice")
            inspect = gr.Button("Check selected voice quality")
            inspect_out = gr.Markdown()
            gr.Markdown("A global default narrator is saved as `data/voices/default_voice.wav`. Your saved personal voice remains available in Dubbing Studio as a secondary source.")

        def do_generate(t, n, desc, text, progress=gr.Progress()):
            line = f"({(desc or '').strip()}) {(text or '').strip()}".strip()
            r = VOX.generate_candidates(t, "voxcpm2", count=int(n), text=line,
                                        progress=lambda m: progress(0.5, desc=str(m)))
            if not r.get("ok"):
                return gr.update(choices=[]), None, r.get("message", "Generation failed.")
            paths = r["paths"]
            choices = [(f"Candidate {i+1}", p) for i, p in enumerate(paths)]
            return gr.update(choices=choices, value=paths[0]), paths[0], r["message"]

        def do_promote(path, text):
            return VOX.set_global_default_voice(path, text)["message"]

        def do_check(name):
            if not name:
                return "Pick a saved voice first."
            from ..common.voicecheck import check_reference
            _ok, msg = check_reference(str(VOICES / name))
            return msg
        
        def do_save_profile(name, t, pers, sd):
            if not name:
                return "Enter a profile name.", gr.update()
            r = VOX.save_narrator_profile(name, t, pers, sd)
            return r["message"], gr.update(choices=VOX.list_narrator_profiles(), value=name)
        
        def do_load_profile(name):
            if not name:
                return "Pick a profile.", "", "", ""
            r = VOX.load_narrator_profile(name)
            if not r.get("ok"):
                return r["message"], "", "", ""
            p = r["profile"]
            return (r["message"], p["target"], p["persona"], p["seed_text"])
        
        def do_del_profile(name):
            if not name:
                return "Pick a profile.", gr.update()
            r = VOX.delete_narrator_profile(name)
            return r["message"], gr.update(choices=VOX.list_narrator_profiles())
        
        def do_rate(path, score, notes):
            if not path:
                return "Select a candidate first."
            r = VOX.rate_candidate(path, float(score), notes or "")
            return r["message"]
        
        def do_consistency(cand, t, progress=gr.Progress()):
            if not cand:
                return "Select a candidate first.", None
            r = VOX.run_consistency_test(t, "voxcpm2", cand, progress=lambda m: progress(0.5, desc=str(m)))
            if not r.get("ok"):
                return r["message"], None
            # Return first sample for quick preview
            return r["message"], r["paths"][0] if r["paths"] else None

        generate.click(do_generate, [target, count, persona, seed], [candidate, player, status])
        candidate.change(lambda p: p or None, candidate, player)
        promote.click(do_promote, [candidate, seed], status)
        inspect.click(do_check, saved, inspect_out)
        
        save_profile.click(do_save_profile, [profile_name, target, persona, seed], [profile_status, profile_pick])
        load_profile.click(do_load_profile, profile_pick, [profile_status, target, persona, seed])
        del_profile.click(do_del_profile, profile_pick, [profile_status, profile_pick])
        rate_btn.click(do_rate, [candidate, cand_score, cand_notes], profile_status)
        consistency_btn.click(do_consistency, [candidate, target], [consistency_status, consistency_player])
    return {}


# =====================================================================
# TAB 3B — Dubbing (5 models, lazy, unload-after)
# =====================================================================
def build_tab3():
    cfg = C.load_config()
    # Server-side state survives Gradio tab navigation/client stream reconnects.
    _dub_jobs: dict[str, dict] = {}
    with gr.Tab("3 · Chatterbox Dubbing"): 
        gr.Markdown("### Generate audio from a forwarded target script")
        with gr.Row():
            proj = gr.Dropdown(H.list_projects(), label="Project")
            target = gr.Dropdown(H.target_choices(), label="Target",
                                 value="hinglish_devanagari")
            refresh = gr.Button("↻")
        with gr.Row():
            model = gr.Dropdown(H.model_choices(), label="Dubbing model",
                                value="indicf5")
            license_note = gr.Markdown()
        gr.Markdown("_Disk-frugal mode: the model **installs on first Dub** (evicting "
                    "other models to fit 10 GB), then its weights are **cleared after "
                    "the dub** to free space for the next model._")
        with gr.Row():
            disk_info = gr.Markdown()
            refresh_disk = gr.Button("↻ Disk status")
            clear_after = gr.Checkbox(True, label="Clear model cache after dub (saves disk)")
        with gr.Row():
            testing_mode = gr.Checkbox(
                False,
                label="🧪 Testing mode — allow ANY model on T4 (ignore VRAM/disk limits, "
                      "e.g. Fish S2 Pro; may OOM)")
            testing_status = gr.Markdown()
        with gr.Row():
            fwd_version = gr.Dropdown([], label="Adaptation version (per language)",
                                      info="Each language keeps its own versions. "
                                           "Pick one, or leave blank for the latest.")
            fwd_refresh = gr.Button("↻ Versions")
            forward_btn = gr.Button("📥 Load selected/latest forwarded script",
                                    variant="primary")
        script_box = gr.Textbox(label="Forwarded narration (editable)", lines=8)
        save_override = gr.Button("Save Dub Override Version")

        # ---- Intro / Outro presets (collapsible; default ON) ----
        with gr.Accordion("🎬 Intro & Outro (adds an opener + closer to every video)",
                          open=False):
            with gr.Row():
                io_enabled = gr.Checkbox(IO.DEFAULT_ENABLED,
                                         label="Add intro & outro (on by default)")
                io_preset = gr.Dropdown(IO.preset_names(), value=IO.DEFAULT_PRESET,
                                        label="Preset")
            _io_def = IO.get_preset(IO.DEFAULT_PRESET)
            io_intro = gr.Textbox(_io_def["intro"], label="Intro line", lines=2)
            io_outro = gr.Textbox(_io_def["outro"], label="Outro line", lines=2)
            with gr.Row():
                io_new_name = gr.Textbox(label="Preset name (to save / add your own)")
                io_save = gr.Button("💾 Save / update preset")
                io_delete = gr.Button("🗑 Delete / revert preset")
            io_status = gr.Markdown()

        with gr.Accordion("Voice controls", open=True):
            with gr.Row():
                voice_mode = gr.Radio(
                    ["Auto default voice (consistent)", "Saved reference voice",
                     "Built-in voice"],
                    value="Auto default voice (consistent)", label="Voice source")
                ref_voice = gr.Dropdown(
                    [p.name for p in VOICES.glob("*.wav")], label="Saved reference voice")
            ref_transcript = gr.Textbox(
                label="Reference transcript (needed for IndicF5 / VoxCPM2)", lines=2)
            gr.Markdown(
                "_**Auto default voice** = generate one voice once & reuse it for the "
                "whole video (consistent, no recording needed). **Saved reference "
                "voice** = clone your own uploaded/recorded voice. **Built-in** = the "
                "model's raw voice (may vary per cue for VoxCPM2)._")
            with gr.Row():
                upload_voice = gr.Audio(
                    label="Upload / record YOUR voice (any length — best ~20-30s auto-used)",
                    type="filepath", sources=["upload", "microphone"])
                upload_voice_name = gr.Textbox(label="Save as (name)", value="my_voice")
                save_voice_btn = gr.Button("💾 Save this voice")
            with gr.Row():
                voice_denoise = gr.Checkbox(
                    False, label="🔇 Clean my reference voice (denoise once when saving)")
                voice_denoise_str = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                              label="Cleaning strength")
            gr.Markdown("_Save once — your voice is stored and reused for every future "
                        "dub (no re-upload). Upload **any length**; the app auto-picks the "
                        "clearest ~25s window (longer references don't clone better and can "
                        "be less consistent). Tick 'Clean my reference voice' to also remove "
                        "background hiss when you save._")
            voice_status = gr.Markdown()

            # ---- Voice Test Lab (collapsible, closed by default) ----
            with gr.Accordion("🔊 Voice test lab — try default voices, save the best, "
                              "delete the rest", open=False):
                with gr.Row():
                    vt_count = gr.Slider(1, 6, value=3, step=1,
                                         label="How many samples to generate")
                    vt_text = gr.Textbox(
                        label="Test line (editable)",
                        value="नमस्ते दोस्तों, चलो कहानी शुरू करते हैं।",
                        placeholder="Leave blank to use the default test line for the target")
                    vt_gen = gr.Button("🎲 Generate voice samples", variant="primary")
                vt_status = gr.Markdown()
                vt_samples = gr.Dropdown([], label="Generated samples (pick to hear)")
                vt_player = gr.Audio(label="Sample preview", interactive=False)
                with gr.Row():
                    vt_save_name = gr.Textbox(label="Save selected sample as", value="voice_1")
                    vt_save = gr.Button("💾 Save selected sample")
                gr.Markdown("**Saved voice library** (reuse for dubbing / delete before you start):")
                with gr.Row():
                    vt_saved = gr.Dropdown([], label="Saved voices (pick to hear / delete)")
                    vt_saved_refresh = gr.Button("↻")
                vt_saved_player = gr.Audio(label="Saved voice preview", interactive=False)
                with gr.Row():
                    vt_use = gr.Button("✅ Use this saved voice for the dub")
                    vt_delete = gr.Button("🗑 Delete selected saved voice", variant="stop")
                vt_manage_status = gr.Markdown()

            with gr.Row():
                instances = gr.Radio(["1", "2", "3", "4", "5"], value="1",
                                     label="TTS instances (parallel; auto-capped by VRAM)")
                force_regen = gr.Checkbox(False, label="Force regenerate (ignore cached cues)")
            with gr.Row():
                denoise_on = gr.Checkbox(
                    False, label="🔇 Denoise dubbed audio (removes slight hiss/hum)")
                denoise_strength = gr.Slider(
                    0.5, 2.0, value=1.0, step=0.1,
                    label="Denoise strength (higher = more aggressive)")
            with gr.Row():
                narrator_speed = gr.Slider(
                    0.5, 2.0, value=1.0, step=0.05,
                    label="🗣 Narrator speed (1.0 = normal · >1 faster · <1 slower · "
                          "pitch kept natural)")
                speed_note = gr.Markdown("")

            def _speed_note(v):
                if v < 0.85 or v > 1.25:
                    return ("_⚠ Outside 0.85×–1.25× the voice can start to sound "
                            "unnatural/rushed — use for testing._")
                return "_Natural range._"
            narrator_speed.change(_speed_note, narrator_speed, speed_note)
            # --- VoxCPM2 FlashAttention 2 + batch (GPU-aware) ---
            _fa2_ok = C.supports_flash_attention()
            _gpu = C.active_gpu_label()
            with gr.Accordion("⚡ VoxCPM2 FlashAttention 2 + batching "
                              "(advanced, sm_80+ GPUs only)", open=False):
                if _fa2_ok:
                    fa2_note = gr.Markdown(
                        f"_Detected **{_gpu}** — supports FlashAttention 2. Enable it "
                        f"and test batch sizes to find the fastest for your clips._")
                else:
                    fa2_note = gr.Markdown(
                        f"_Detected **{_gpu}**. FlashAttention 2 needs an Ampere+ GPU "
                        f"(sm_80). A T4 is sm_75, so FA2 + batching are **unavailable "
                        f"here** and run at 1×. Switch to an L4 to use this. "
                        f"(Verified: FA2 does not support Turing.)_")
                with gr.Row():
                    fa2_on = gr.Checkbox(
                        _fa2_ok, interactive=_fa2_ok,
                        label="Use FlashAttention 2 for VoxCPM2")
                    vox_batch = gr.Slider(
                        1, 8, value=4 if _fa2_ok else 1, step=1,
                        interactive=_fa2_ok,
                        label="VoxCPM2 batch size (concurrent cues; sm_80+ only)")
                energy = gr.Dropdown(["natural", "expressive", "calm"],
                                     value="expressive", label="Narration energy")
                emotion_tags = gr.Textbox(
                    label="Emotion / style hint",
                    placeholder="Fish: [excited]  ·  VoxCPM2: (energetic, fast)  ·  "
                                "Qwen3-TTS: Very happy.  (leave blank for neutral)")
            with gr.Accordion("🎛 Advanced TTS controls (fine-tune voice; optional)",
                              open=False):
                gr.Markdown("_The **Narration energy** preset above already sets good "
                            "values. Override them here only if you want to fine-tune. "
                            "0 = use the preset's value._")
                with gr.Row():
                    adv_exaggeration = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="Exaggeration (Chatterbox expressiveness; 0 = preset)")
                    adv_cfg = gr.Slider(
                        0.0, 5.0, value=0.0, step=0.1,
                        label="Guidance / cfg (VoxCPM2 cfg_value; 0 = preset)")
                with gr.Row():
                    adv_temperature = gr.Slider(
                        0.0, 1.5, value=0.0, step=0.05,
                        label="Temperature (variation; 0 = preset)")
                    adv_steps = gr.Slider(
                        0, 30, value=0, step=1,
                        label="Inference steps (VoxCPM2 quality/speed; 0 = preset)")
            
            # VoxCPM2 Clone Mode selection (controls narrator identity in long dubs)
            with gr.Accordion("🔀 VoxCPM2 Clone Mode (identity vs emotion control)", open=False):
                gr.Markdown("**Hi-Fi Clone** = prompt_wav + prompt_text + reference_wav → highest voice similarity, style prefix ignored.\n"
                            "**Controllable Clone** = reference_wav + style prefix → emotion/style controllable.\n"
                            "**Hybrid Smart Routing** = first cue Hi-Fi (if reference transcript provided), then Controllable for rest.")
                clone_mode = gr.Radio(
                    ["Hybrid Smart Routing (recommended)", "Hi-Fi Clone (max fidelity)", "Controllable Clone (style control)"],
                    value="Hybrid Smart Routing (recommended)",
                    label="Clone mode for VoxCPM2"
                )
                gr.Markdown("_Hi-Fi needs reference transcript for exact cloning. Controllable uses your style prefix. "
                            "Hybrid uses Hi-Fi for cue 1 if transcript is given, then Controllable for consistency._")
        with gr.Accordion("Live Render Pipeline (optional)", open=False):
            live_on = gr.Checkbox(False, label="Build cue-locked video groups while TTS runs")
            with gr.Row():
                group_size = gr.Slider(4, 24, value=12, step=1, label="Cue group size")
                vram_reserve = gr.Slider(1, 6, value=2, step=1, label="Min free VRAM (GB)")
            with gr.Row():
                pause_btn = gr.Button("Pause")
                resume_btn = gr.Button("Resume")
                cancel_btn = gr.Button("Cancel")
                dash_auto = gr.Checkbox(True, label="Auto-refresh dashboard")
            dashboard = gr.JSON(label="Live dashboard")
            dash_timer = gr.Timer(2.0)   # BUG FIX: live dashboard now auto-updates
        with gr.Accordion("⚡ One-Click Auto (Dub → Render → Final MP4)", open=True):
            gr.Markdown("_Press once. Runs TTS + parallel video render + merge and gives "
                        "you the finished dubbed MP4. Extras are optional (default off for speed)._")
            with gr.Row():
                auto_captions = gr.Checkbox(False, label="Burn target captions")
                auto_mask = gr.Checkbox(False, label="Hide Chinese subtitles (mask)")
                auto_bgm = gr.Checkbox(False, label="Add BGM")
            with gr.Accordion("🎭 Mask settings (when 'Hide Chinese subtitles' is on)",
                              open=False):
                with gr.Row():
                    auto_mask_type = gr.Dropdown(MASK_TYPES, value="Blur + dark band",
                                                 label="Blur / mask type")
                    auto_mask_color = gr.Dropdown(MASK_COLORS, value="black",
                                                  label="Band / cover color")
                with gr.Row():
                    mask_x = gr.Slider(0, 3840, value=308, step=2, label="Mask X")
                    mask_y = gr.Slider(0, 2160, value=946, step=2, label="Mask Y")
                with gr.Row():
                    mask_w = gr.Slider(2, 3840, value=854, step=2, label="Mask W")
                    mask_h = gr.Slider(2, 2160, value=90, step=2, label="Mask H")
                with gr.Row():
                    auto_mask_strength = gr.Slider(1, 40, value=10, step=1,
                                                   label="Blur/pixelate strength")
                    auto_mask_opacity = gr.Slider(0, 1, value=0.6, step=0.05,
                                                  label="Band opacity")
                    auto_mask_ptime = gr.Slider(0, 600, value=5, step=1,
                                                label="Preview time (s)")
                auto_mask_preview_btn = gr.Button("👁 Preview mask on a real frame")
                auto_mask_preview = gr.Image(label="Mask preview")
            with gr.Accordion("🎵 BGM (when 'Add BGM' is on)", open=False):
                with gr.Row():
                    auto_bgm_file = gr.Dropdown([p.name for p in BGM.glob("*")],
                                                label="BGM file (from data/bgm/)")
                    auto_bgm_refresh = gr.Button("↻")
                with gr.Row():
                    auto_bgm_upload = gr.Audio(label="…or upload a BGM track",
                                               type="filepath", sources=["upload"])
                    auto_bgm_name = gr.Textbox(label="Save as", value="my_bgm")
                    auto_bgm_save = gr.Button("💾 Save BGM")
                auto_bgm_status = gr.Markdown()
            auto_btn = gr.Button("⚡ Auto: Dub → Final Dubbed MP4", variant="primary")
            auto_video = gr.Video(label="Finished dubbed video")
            with gr.Row():
                auto_mp4 = gr.File(label="MP4")
                auto_srt = gr.File(label="SRT")
                auto_script = gr.File(label="Script")
                auto_quality = gr.File(label="Quality JSON")
        gr.Markdown("— or generate cues only (then export in Tab 5) —")
        with gr.Row():
            dub_btn = gr.Button("▶ Dub (generate cues only)", variant="secondary")
            cancel_dub_btn = gr.Button("🛑 Cancel current dub", variant="stop")
        progress = gr.Markdown()
        with gr.Accordion("📊 Persistent Dub Status", open=True):
            dub_dashboard = gr.JSON(label="Current server-side job state")
            dub_dashboard_timer = gr.Timer(2.0)

        def do_refresh():
            return (gr.update(choices=H.list_projects()),
                    gr.update(choices=[p.name for p in VOICES.glob("*.wav")]))

        def show_license(mid):
            m = C.model_cfg(mid)
            flag = m.get("license_flag", "")
            warn = ""
            if mid == "fish":
                warn = ("\n\n⚠ **Fish S2 Pro self-host for MONETIZED/commercial use requires a "
                        "PAID license from Fish Audio.** Free for research/non-commercial only.")
            gated = "  🔒 gated download (needs HF token)" if m.get("weights_gated") else ""
            return f"**License:** {flag}{gated}{warn}"

        def _fwd_version_choices(p, t):
            """Populate the version dropdown for THIS project+language."""
            if not p:
                return gr.update(choices=[], value=None)
            dets = PKG.version_details(safe_name(p), t)
            # newest first, labelled with time/lines/model so you know which is which
            choices = [f"{d['version']} · {d['created']} · {d['lines']} lines · {d['model']}"
                       + (f" (override of {d['override_of']})" if d['override_of'] else "")
                       for d in reversed(dets)]
            return gr.update(choices=choices, value=(choices[0] if choices else None))

        def do_forward(p, t, ver_label):
            # returns (script_text, model_dropdown_update, version_dd_update, status)
            if not p:
                return "", gr.update(), gr.update(), "Pick a project."
            # extract "V<n>" from the dropdown label; blank -> latest
            version = None
            if ver_label:
                version = str(ver_label).split(" ")[0].strip() or None
            pkg = PKG.load_package(safe_name(p), t, version=version)
            if not pkg:
                return ("", gr.update(), _fwd_version_choices(p, t),
                        f"No forwarded package for **{t}**. Forward a script in Tab 2 first "
                        f"(each language has its own versions).")
            text = "\n".join(pkg.get("narration_lines", []))
            chosen = pkg.get("dubbing_model")
            upd = gr.update(value=chosen) if chosen else gr.update()
            note = (f"✅ Loaded **{pkg.get('_version','?')}** for **{t}** "
                    f"({len(pkg.get('narration_lines', []))} lines"
                    + (f", model: {C.model_cfg(chosen)['label']}" if chosen else "") + ").")
            return text, upd, _fwd_version_choices(p, t), note

        def do_disk():
            from ..common.diskmanager import disk_free_gb, installed_model_venvs
            return (f"**Disk free:** {disk_free_gb():.1f} GB · "
                    f"**installed models:** {', '.join(installed_model_venvs()) or 'none'}")

        def do_dub(p, t, mid, text, vmode, refv, reftext, energy_v, tags, live, gs,
                   clr, inst, fregen, dn_on, dn_strength, fa2, vbatch,
                   io_on, io_in, io_out, nar_speed,
                   adv_exag, adv_cfg_v, adv_temp, adv_step,
                   clone_mode,
                   progress=gr.Progress()):
            if not p:
                yield "Pick a project."; return
            pid = safe_name(p)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                yield "No narration lines."; return
            # Intro/outro: add opener + closer lines (spoken over first/last frames).
            lines = IO.apply_to_lines(lines, io_in, io_out, enabled=bool(io_on))
            # Narrator speed part 2: for models that read style text, add a native
            # pace hint to the emotion tag (VoxCPM2/Fish). The pitch-preserving
            # atempo slider (part 1) still applies to every model afterwards.
            from ..adapt import emotions as _EMO
            _sh = _EMO.speed_hint(mid, float(nar_speed))
            if _sh:
                tags = (f"{tags} {_sh}".strip() if tags else _sh)
            from ..common.diskmanager import fits_budget
            okd, dmsg = fits_budget(mid)
            if not okd:
                yield f"❌ Disk: {dmsg}"; return
            chk = check_model_fits(mid, live_render=live)
            if not chk.ok:
                yield f"❌ {chk.warning}"; return
            # Instance clamp note (e.g. VoxCPM2 -> 1 on a 16 GB T4). Never blocks.
            from ..dubbing.router import instance_cap_note
            _cap = instance_cap_note(mid, int(inst))
            if _cap:
                yield _cap
            # VoxCPM2 FA2 + batch: only honored on sm_80+ GPUs (verified). On T4
            # these are ignored (control is disabled), so no crash can occur.
            import os as _os
            if mid == "voxcpm2" and C.supports_flash_attention():
                _os.environ["VOXCPM_FLASH_ATTN"] = "1" if fa2 else "0"
                _os.environ["VOXCPM_BATCH_SIZE"] = str(int(vbatch or 1))
            else:
                _os.environ.pop("VOXCPM_FLASH_ATTN", None)
                _os.environ["VOXCPM_BATCH_SIZE"] = "1"
            mcfg = C.model_cfg(mid)
            reference = None
            if vmode == "Saved reference voice" and refv:
                reference = str(VOICES / refv)
            elif vmode == "Auto default voice (consistent)":
                # Generate ONE voice once and reuse it -> consistent voice, no upload.
                from ..directaudio import voices as VOX
                yield "⏳ Preparing a consistent default voice (one-time)…"
                dv = VOX.ensure_default_voice(t, mid)
                if not dv.get("ok"):
                    yield f"❌ {dv.get('message')}"; return
                reference = dv["path"]
                if not (reftext or "").strip():
                    reftext = dv["text"]   # auto transcript for the default voice
            # Models that need a reference (VoxCPM2/IndicF5): block only if 'Built-in'
            # was explicitly chosen with no reference (produces inconsistent voice).
            if mcfg.get("needs_ref_transcript") and not reference:
                yield (f"❌ {mcfg['label']} needs a reference voice. Use 'Auto default "
                       f"voice (consistent)' (easiest), or 'Saved reference voice' with "
                       f"a saved/uploaded voice + its transcript."); return
            if mcfg.get("needs_ref_transcript") and reference and not (reftext or "").strip():
                yield (f"⚠ {mcfg['label']} works best with a reference transcript. "
                       f"Continuing without one…")
            # Strip emotion tags if the chosen model can't read them (never spoken aloud)
            from ..adapt import emotions as EMO
            lines, n_stripped = EMO.strip_tags_if_incapable(mid, lines)
            progress(0.02, desc="Preparing dub…")
            strip_note = (f" (removed emotion tags from {n_stripped} line(s) — "
                          f"{mcfg['label']} can't read them)" if n_stripped else "")
            yield f"⏳ Preparing dub… (model installs on first run; this can take a while){strip_note}"
            reqs = []
            cue_dir = edition_dir(pid, t) / "tts_cues"
            cue_dir.mkdir(parents=True, exist_ok=True)
            from ..dubbing.workers.protocol import GenRequest
            try:
                from ..dubbing import cleanup
            except ModuleNotFoundError as e:
                yield (f"❌ Missing Python package '{e.name}' in the app environment. "
                       f"Run:  pip install soundfile numpy  (then restart the app)."); return
            preset = dict(C.load_config()["tts_quality"]["presets"][energy_v])
            # Advanced overrides (0 = keep preset value). Model workers read the
            # keys they understand and ignore the rest.
            if float(adv_exag) > 0:
                preset["exaggeration"] = float(adv_exag)
            if float(adv_cfg_v) > 0:
                preset["cfg_value"] = float(adv_cfg_v); preset["cfg_weight"] = float(adv_cfg_v)
            if float(adv_temp) > 0:
                preset["temperature"] = float(adv_temp)
            if int(adv_step) > 0:
                preset["inference_timesteps"] = int(adv_step)
            for i, ln in enumerate(lines):
                raw = cue_dir / f"cue_{i:04d}_raw.wav"
                reqs.append(GenRequest(
                    text=ln, out_path=str(raw), target=t,
                    language=TARGET_LANG.get(t, "en"),
                    reference_wav=reference, reference_text=reftext or None,
                    preset=preset, emotion_tags=tags or None).to_json())
            warn_prefix = ""
            if chk.warning:
                warn_prefix = f"⚠ {chk.warning}\n\n"

            ok_count = {"n": 0}

            # ----- OPTIONAL: TRUE PARALLEL LIVE RENDER -----
            pipe = None
            cue_objs = []
            if live:
                import json as _json
                from ..export import timeline as TL, exporter as EX
                from ..dubbing.live_render import LivePipeline
                trj = project_dir(pid) / "transcript" / "transcript.json"
                _srcv = find_source_video(pid)
                srcs = [_srcv] if _srcv else []
                if trj.exists() and srcs:
                    tr = _json.loads(trj.read_text(encoding="utf-8"))
                    for i in range(len(lines)):
                        seg = tr[i] if i < len(tr) else {"start": 0, "end": 1}
                        cue_objs.append(TL.Cue(idx=i, src_start=seg["start"],
                                               src_end=seg["end"], text=lines[i]))
                    pipe = LivePipeline(pid, t)
                    _live["pipes"][_pipe_key(p, t)] = pipe
                    src_video = str(srcs[0])
                    work = edition_dir(pid, t) / "live_render_groups" / "work"
                    work.mkdir(parents=True, exist_ok=True)

                    def render_group(g, group_cues):
                        # fill each cue's audio_seconds from its cleaned WAV, no-gap timeline
                        import soundfile as sf
                        for c in group_cues:
                            wav = cue_dir / f"cue_{c.idx:04d}.wav"
                            if wav.exists():
                                info = sf.info(str(wav))
                                c.audio_seconds = info.frames / info.samplerate
                        sub_tl = TL.build_cue_locked(group_cues)  # NO GAPS
                        # keep groups uniformly encoded (see auto_pipeline note)
                        lst = EX.build_segments_concat(src_video, sub_tl, work / f"g{g}",
                                                       fast_copy=False)
                        out = pipe.groups_dir / f"group_{g:03d}.mp4"
                        EX.concat_video(lst, out, reencode=True)  # clean timeline (no freeze)
                        return out

                    pipe.start(cue_objs, render_group)

            # ----- TTS generation (continuous pipeline; cleanup overlaps GPU) -----
            def post_process(i, r):
                if r.get("ok"):
                    raw = cue_dir / f"cue_{i:04d}_raw.wav"
                    clean = cue_dir / f"cue_{i:04d}.wav"
                    try:
                        if r.get("skipped") and clean.exists():
                            pass
                        elif raw.exists():
                            cleanup.clean_cue(str(raw), str(clean),
                                              denoise=bool(dn_on),
                                              denoise_strength=float(dn_strength),
                                              speed=float(nar_speed))
                        if clean.exists():
                            ok_count["n"] += 1
                            if pipe:
                                pipe.mark_cue_ready(i)
                    except Exception as e:
                        from ..common.logging_util import get_logger
                        get_logger("ui.dub").warning("cue %s cleanup failed: %s", i, e)

            def on_cue(i, r):
                n = ok_count["n"]
                if job_id in _dub_jobs:
                    _dub_jobs[job_id].update({"done": n, "current_cue": i + 1,
                                              "message": "Cue completed" if r.get("ok") else r.get("error", "Cue failed")})
                frac = 0.05 + 0.9 * (n / max(1, len(lines)))
                try:
                    progress(min(0.95, frac), desc=f"TTS {n}/{len(lines)} cues done…")
                except Exception:
                    pass

            # Cancel support: register an event keyed by project+target so the
            # separate "Cancel Dub" button (different request thread) can stop it.
            job_id = f"{pid}::{t}"
            cancel_ev = get_router().make_cancel_event(job_id)
            _dub_jobs[job_id] = {"state": "running", "project": pid, "target": t,
                                 "model": mid, "done": 0, "total": len(lines),
                                 "current_cue": None, "message": "Starting TTS…"}

            results = get_router().generate_stream(
                mid, reqs, instances=int(inst), post_process=post_process,
                on_cue=on_cue,
                clear_cache_after=clr, keep_venv=True,
                force_regenerate=bool(fregen),
                cancel_event=cancel_ev)
            get_router().clear_cancel_event(job_id)
            was_cancelled = cancel_ev.is_set()
            if job_id in _dub_jobs:
                _dub_jobs[job_id].update({"state": "cancelled" if was_cancelled else "finishing",
                                          "message": "Cancelled" if was_cancelled else "Finishing render/export"})

            if pipe:
                if was_cancelled:
                    try:
                        pipe.cancel()
                    except Exception:
                        pass
                else:
                    progress(0.96, desc="Finishing live video render…")
                    yield f"✅ TTS done ({ok_count['n']}/{len(lines)}). Finishing live render…"
                    pipe.mark_tts_done()
                    # BUG FIX: don't block for up to an hour in one call (looks frozen).
                    # Poll the pipeline and keep yielding live progress until it finishes.
                    import time as _t
                    deadline = _t.time() + 7200
                    while _t.time() < deadline:
                        snap = pipe.snapshot()
                        done_g = snap.get("completed_groups", 0)
                        tot_g = snap.get("total_groups", 0) or 1
                        state = snap.get("pipeline_state", "running")
                        progress(min(0.99, 0.96 + 0.03 * done_g / tot_g),
                                 desc=f"Rendering video groups {done_g}/{tot_g}…")
                        yield (f"🎬 Live render: {done_g}/{tot_g} groups done "
                               f"({state})… TTS {ok_count['n']}/{len(lines)} cues.")
                        if state in ("done", "cancelled", "error") or \
                                (tot_g and done_g >= tot_g):
                            break
                        _t.sleep(2)

            ok = ok_count["n"]
            if job_id in _dub_jobs:
                _dub_jobs[job_id].update({"state": "cancelled" if was_cancelled else "completed",
                                          "done": ok, "message": "Dub cancelled" if was_cancelled else "Dub completed"})
            if ok == 0 and results:
                err = results[0].get("error", "unknown error") if results else "no result"
                tr = results[0].get("trace") if results else None
                msg = (f"❌ Dub failed: {err}\n\n"
                       f"📋 Full details are in the **Live Logs** tab (Tab 7) — look for "
                       f"'cue 0 worker traceback' / 'worker log tail'.")
                if tr:
                    msg += f"\n\n```\n{tr[-1500:]}\n```"
                yield msg; return
            from ..common.diskmanager import disk_free_gb
            tail = (" Model cache cleared." if clr else "")
            live_note = ""
            if pipe:
                snap = pipe.snapshot()
                live_note = (f" Live render: {snap['completed_groups']}/"
                             f"{snap['total_groups']} groups done (parallel).")
            if ok > 0 and not was_cancelled:
                from ..common import stageflow as SF
                SF.mark_stage(pid, "dubbing")
            progress(1.0, desc="Done")
            if was_cancelled:
                yield (f"🛑 Dub cancelled — kept {ok}/{len(lines)} cues already done. "
                       f"Press Dub again to resume (finished cues are skipped). "
                       f"Disk free: {disk_free_gb():.1f} GB.")
            else:
                yield (f"{warn_prefix}✅ Dubbed {ok}/{len(lines)} cues with "
                       f"{C.model_cfg(mid)['label']}.{tail}{live_note} "
                       f"Disk free: {disk_free_gb():.1f} GB.")

        refresh.click(do_refresh, outputs=[proj, ref_voice])

        def do_save_voice(fp, nm, dn, dn_str, src_lang_v):
            from ..directaudio import voices as VOX
            # tell it which TTS model is loaded so Whisper can stay CO-RESIDENT
            # (your flow: TTS idle in VRAM, Whisper loads beside it, transcribes,
            # unloads, TTS untouched). If VRAM is tight it evicts + we reload below.
            loaded = None
            try:
                loaded = get_router().current_model()
            except Exception:
                loaded = None
            r = VOX.save_uploaded_voice(fp, nm, denoise=bool(dn),
                                        denoise_strength=float(dn_str),
                                        auto_transcribe=True,
                                        tts_loaded_model=loaded,
                                        source_language=src_lang_v or "Auto")
            # if Whisper had to evict TTS to fit, reload it so you continue instantly
            if r.get("freed_tts") and loaded:
                try:
                    get_router().load(loaded)
                except Exception:
                    pass
            choices = [p.name for p in VOICES.glob("*.wav")]
            if r.get("ok"):
                return (r["message"], gr.update(choices=choices, value=r["name"]),
                        gr.update(value="Saved reference voice"))
            return r["message"], gr.update(choices=choices), gr.update()

        save_voice_btn.click(do_save_voice,
                             [upload_voice, upload_voice_name, voice_denoise,
                              voice_denoise_str, gr.State("Auto")],
                             [voice_status, ref_voice, voice_mode])

        # ---- Voice Test Lab handlers ----
        from ..directaudio import voices as VOX

        def _autofill_ref_transcript(name):
            """When you pick a saved voice, auto-load its transcript sidecar (if any)
            so VoxCPM2 clones at full fidelity with no typing."""
            tr = VOX.transcript_for_voice(name) if name else ""
            return gr.update(value=tr) if tr else gr.update()
        ref_voice.change(_autofill_ref_transcript, ref_voice, ref_transcript)

        def _saved_choices():
            return gr.update(choices=VOX.list_voices())

        def do_vt_generate(p, t, mid, n, text, progress=gr.Progress()):
            # p (project) not required to test a voice; model + target are.
            r = VOX.generate_candidates(t, mid, count=int(n), text=text,
                                        progress=lambda m: progress(0.5, desc=str(m)))
            if not r.get("ok"):
                return gr.update(choices=[]), None, f"❌ {r.get('message')}"
            paths = r["paths"]
            # label samples 1..N; value carries the full path so we can play it
            choices = [(f"Sample {i+1}", pth) for i, pth in enumerate(paths)]
            first = paths[0]
            return (gr.update(choices=choices, value=first), first, r["message"])

        def do_vt_play(path):
            return path if path else None

        def do_vt_save(cand_path, nm, tline):
            # the lab KNOWS the transcript (it's the test line) -> store it so the
            # dub can clone at full fidelity without you typing anything.
            r = VOX.save_candidate(cand_path, nm, transcript=tline or "")
            saved = VOX.list_voices()
            if r.get("ok"):
                # add to both the test-lab saved list AND the main reference dropdown
                return (gr.update(choices=saved, value=r["name"]),
                        gr.update(choices=saved, value=r["name"]),
                        r["message"])
            return gr.update(choices=saved), gr.update(choices=saved), r["message"]

        def do_vt_saved_play(name):
            if not name:
                return None
            p = VOICES / name
            return str(p) if p.exists() else None

        def do_vt_use(name):
            if not name:
                return gr.update(), gr.update(), gr.update(), "Pick a saved voice first."
            # auto-fill the reference transcript from the sidecar (full-fidelity clone)
            tr = VOX.transcript_for_voice(name)
            note = (f"✅ Using '{name}' for the dub."
                    + (" Transcript auto-filled for best quality."
                       if tr else " (audio-only clone — no transcript needed.)"))
            return (gr.update(value="Saved reference voice"), gr.update(value=name),
                    gr.update(value=tr) if tr else gr.update(), note)

        def do_vt_delete(name):
            r = VOX.delete_voice(name)
            saved = VOX.list_voices()
            newsel = saved[0] if saved else None
            return (gr.update(choices=saved, value=newsel),   # test-lab list
                    gr.update(choices=saved),                 # main ref dropdown
                    None,                                     # clear player
                    r["message"])

        vt_gen.click(do_vt_generate, [proj, target, model, vt_count, vt_text],
                     [vt_samples, vt_player, vt_status])
        vt_samples.change(do_vt_play, vt_samples, vt_player)
        vt_save.click(do_vt_save, [vt_samples, vt_save_name, vt_text],
                      [vt_saved, ref_voice, vt_manage_status])
        vt_saved_refresh.click(lambda: (_saved_choices()), outputs=vt_saved)
        vt_saved.change(do_vt_saved_play, vt_saved, vt_saved_player)
        vt_use.click(do_vt_use, vt_saved,
                     [voice_mode, ref_voice, ref_transcript, vt_manage_status])
        vt_delete.click(do_vt_delete, vt_saved,
                        [vt_saved, ref_voice, vt_saved_player, vt_manage_status])

        refresh_disk.click(do_disk, outputs=disk_info)

        def do_testing_toggle(on):
            from ..common.diskmanager import set_testing_mode
            set_testing_mode(bool(on))
            if on:
                return ("🧪 **Testing mode ON** — every model can load on this GPU "
                        "(disk/VRAM/instance limits bypassed). Big models like Fish "
                        "S2 Pro may run out of memory; that's expected while testing.")
            return "Testing mode off — normal safety limits restored."
        testing_mode.change(do_testing_toggle, testing_mode, testing_status)
        model.change(show_license, model, license_note)
        # U-3: live-render pipeline handles keyed by (project, target) so two
        # projects (or two users on a shared link) never control each other's dub.
        _live = {"pipes": {}}

        def _pipe_key(p, t):
            return f"{safe_name(p)}::{t}" if p else None

        def _get_pipe(p, t):
            """Return the EXISTING pipeline for this project/target (do NOT create a
            fresh one — a new handle wouldn't match the running dub's threads)."""
            return _live["pipes"].get(_pipe_key(p, t))

        def do_live_pause(p, t):
            pipe = _get_pipe(p, t)
            if pipe:
                pipe.pause()
                return pipe.snapshot()
            return {"state": "no pipeline"}

        def do_live_resume(p, t):
            pipe = _get_pipe(p, t)
            if pipe:
                pipe.resume()
                return pipe.snapshot()
            return {"state": "no pipeline"}

        def do_live_cancel(p, t):
            pipe = _get_pipe(p, t)
            if pipe:
                pipe.cancel()
                return pipe.snapshot()
            return {"state": "no pipeline"}

        def do_save_override(p, t, text):
            if not p:
                return "Pick a project."
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return "Nothing to save."
            versions = PKG.list_versions(safe_name(p), t)
            base = versions[-1] if versions else "V1"
            ver = PKG.save_override(safe_name(p), t, base,
                                    {"target": t, "narration_lines": lines,
                                     "source": "dub_override"})
            return f"Saved Dub Override Version {ver} (original adaptation kept)."

        def do_auto(p, t, mid, text, vmode, refv, reftext, energy_v, tags,
                    a_caps, a_mask, a_bgm, a_bgm_file, clr, inst, fregen,
                    m_x, m_y, m_w, m_h, m_type, m_color, m_strength, m_opacity,
                    io_on, io_in, io_out, nar_speed,
                    progress=gr.Progress()):
            if not p:
                yield None, None, None, None, None, "Pick a project."; return
            pid = safe_name(p)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                yield None, None, None, None, None, \
                    "No narration lines (load a forwarded script)."; return
            # Intro/outro opener + closer (spoken over first/last frames).
            lines = IO.apply_to_lines(lines, io_in, io_out, enabled=bool(io_on))
            # Narrator speed: native pace hint for capable models (slider does the rest).
            from ..adapt import emotions as _EMO
            _sh = _EMO.speed_hint(mid, float(nar_speed))
            if _sh:
                tags = (f"{tags} {_sh}".strip() if tags else _sh)
            mcfg = C.model_cfg(mid)
            reference = None
            if vmode == "Saved reference voice" and refv:
                reference = str(VOICES / refv)
            elif vmode == "Auto default voice (consistent)":
                from ..directaudio import voices as VOX
                dv = VOX.ensure_default_voice(t, mid)
                if not dv.get("ok"):
                    yield None, None, None, None, None, f"❌ {dv.get('message')}"; return
                reference = dv["path"]
                if not (reftext or "").strip():
                    reftext = dv["text"]
            if mcfg.get("needs_ref_transcript") and not reference:
                yield None, None, None, None, None, (
                    f"❌ {mcfg['label']} needs a reference voice. Use 'Auto default voice "
                    f"(consistent)' or 'Saved reference voice'."); return
            # Strip emotion tags if the chosen model can't read them (never spoken aloud).
            from ..adapt import emotions as EMO
            lines, _n = EMO.strip_tags_if_incapable(mid, lines)
            # Mask region + TYPE + COLOR + strength all come from the UI now
            # (previously type/strength were hardcoded, so choices had no effect).
            mask_opts = {"type": m_type or "Blur + dark band", "color": m_color or "black",
                         "x": int(m_x), "y": int(m_y), "w": int(m_w), "h": int(m_h),
                         "strength": int(m_strength), "opacity": float(m_opacity)}
            bgm_path = str(BGM / a_bgm_file) if (a_bgm and a_bgm_file) else None

            # U-1: run the (blocking) pipeline in a thread and stream its progress msgs.
            import threading, queue as _q
            msgq: "_q.Queue" = _q.Queue()
            result = {}
            # Auto used to omit this registration, making the shared Cancel button
            # report success while an Auto dub continued running.
            job_id = f"{pid}::{t}"
            cancel_ev = get_router().make_cancel_event(job_id)
            _dub_jobs[job_id] = {"state": "running", "project": pid, "target": t,
                                 "model": mid, "done": 0, "total": len(lines),
                                 "current_cue": None, "message": "Starting Auto pipeline…"}

            def _cb(msg):
                message = str(msg)
                if job_id in _dub_jobs:
                    _dub_jobs[job_id]["message"] = message
                    import re as _re
                    m = _re.search(r"TTS\s+(\d+)\s*/\s*(\d+)", message)
                    if m:
                        _dub_jobs[job_id].update({"done": int(m.group(1)), "total": int(m.group(2))})
                msgq.put(message)

            def _work():
                try:
                    from ..dubbing.auto_pipeline import run_auto
                except ModuleNotFoundError as e:
                    result["r"] = {"ok": False, "message": (
                        f"Missing Python package '{e.name}' in the app environment. "
                        f"Install it into the venv running the app, e.g.:  "
                        f"pip install soundfile numpy  (then restart the app).")}
                    msgq.put(None); return
                try:
                    result["r"] = run_auto(
                        pid, t, mid, lines, energy=energy_v,
                        reference_wav=reference, reference_text=reftext or None,
                        emotion_tags=tags or None, use_live=True,
                        clear_cache_after=clr, captions=a_caps, mask=a_mask,
                        mask_opts=mask_opts if a_mask else None, bgm_path=bgm_path,
                        instances=int(inst), force_regenerate=bool(fregen),
                        narrator_speed=float(nar_speed),
                        intro_added=bool(io_on and (io_in or "").strip()),
                        outro_added=bool(io_on and (io_out or "").strip()),
                        cancel_event=cancel_ev, progress=_cb)
                except Exception as e:  # surface, never hang
                    result["r"] = {"ok": False, "message": f"pipeline error: {e}"}
                finally:
                    get_router().clear_cancel_event(job_id)
                    msgq.put(None)  # sentinel = done

            th = threading.Thread(target=_work, daemon=True)
            th.start()
            progress(0.05, desc="Starting Auto pipeline…")
            last = "⏳ Starting…"
            while True:
                msg = msgq.get()
                if msg is None:
                    break
                last = msg
                progress(0.5, desc=msg[:80])
                yield None, None, None, None, None, msg
            th.join(timeout=5)

            r = result.get("r") or {"ok": False, "message": "no result"}
            if job_id in _dub_jobs:
                _dub_jobs[job_id].update({"state": "completed" if r.get("ok") else ("cancelled" if r.get("cancelled") else "failed"),
                                          "message": r.get("message", "Auto pipeline finished")})
            if not r.get("ok"):
                yield None, None, None, None, None, f"❌ {r.get('message')}"; return
            progress(1.0, desc="Done")
            yield (r["final"], r["final"], r.get("srt"), r.get("script"),
                   r.get("quality"),
                   f"✅ Finished: {r['cues']} cues, {r['seconds']:.0f}s video ready.")

        auto_btn.click(do_auto,
                       [proj, target, model, script_box, voice_mode, ref_voice,
                        ref_transcript, energy, emotion_tags,
                        auto_captions, auto_mask, auto_bgm, auto_bgm_file, clear_after,
                        instances, force_regen, mask_x, mask_y, mask_w, mask_h,
                        auto_mask_type, auto_mask_color, auto_mask_strength, auto_mask_opacity,
                        io_enabled, io_intro, io_outro, narrator_speed],
                       [auto_video, auto_mp4, auto_srt, auto_script, auto_quality, progress])

        # --- One-Click Auto: BGM save/refresh + mask preview (so you CAN select) ---
        def do_auto_bgm_save(upath, name):
            if not upath:
                return gr.update(), "Upload a BGM file first."
            import shutil as _sh
            from pathlib import Path as _P
            BGM.mkdir(parents=True, exist_ok=True)
            ext = _P(upath).suffix or ".mp3"
            dst = BGM / f"{safe_name(name or 'my_bgm')}{ext}"
            _sh.copy(upath, dst)
            return (gr.update(choices=[p.name for p in BGM.glob("*")], value=dst.name),
                    f"✅ Saved **{dst.name}** and selected it. Tick 'Add BGM' to use it.")

        def do_auto_bgm_refresh():
            return gr.update(choices=[p.name for p in BGM.glob("*")])

        def do_auto_mask_preview(p, t, mtype, x, y, w, h, strength_v, opacity_v, ptime, color_v):
            if not p:
                return None
            import subprocess
            pid = safe_name(p)
            srcv = find_source_video(pid)
            if not srcv:
                return None
            from ..export.subtitle_mask import build_mask_filter
            outdir = edition_dir(pid, t) / "exports" / "V1"; outdir.mkdir(parents=True, exist_ok=True)
            frame = outdir / "auto_mask_preview.png"
            fc = build_mask_filter(mtype, x, y, w, h, int(strength_v), float(opacity_v),
                                   color=color_v)
            try:
                subprocess.run(["ffmpeg", "-y", "-ss", str(ptime), "-i", str(srcv),
                                "-filter_complex", fc, "-map", "[v]", "-frames:v", "1",
                                str(frame)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                return str(frame)
            except Exception as e:  # noqa: BLE001
                from ..common.logging_util import get_logger
                get_logger("ui.auto").warning("auto mask preview failed: %s", e)
                return None

        auto_bgm_save.click(do_auto_bgm_save, [auto_bgm_upload, auto_bgm_name],
                            [auto_bgm_file, auto_bgm_status])
        auto_bgm_refresh.click(do_auto_bgm_refresh, outputs=auto_bgm_file)
        auto_mask_preview_btn.click(
            do_auto_mask_preview,
            [proj, target, auto_mask_type, mask_x, mask_y, mask_w, mask_h,
             auto_mask_strength, auto_mask_opacity, auto_mask_ptime, auto_mask_color],
            auto_mask_preview)
        forward_btn.click(do_forward, [proj, target, fwd_version],
                          [script_box, model, fwd_version, progress])
        fwd_refresh.click(_fwd_version_choices, [proj, target], fwd_version)
        # auto-list versions when you switch project or language (so you always see
        # the right per-language history and can access previous adaptations)
        proj.change(_fwd_version_choices, [proj, target], fwd_version)
        target.change(_fwd_version_choices, [proj, target], fwd_version)
        save_override.click(do_save_override, [proj, target, script_box], progress)

        # ---- Intro/Outro preset handlers ----
        def _io_fill(name):
            p = IO.get_preset(name)
            return p.get("intro", ""), p.get("outro", "")

        def _io_save(name, intro, outro):
            # if no explicit new name, edit the currently-selected preset
            target_name = (name or "").strip() or io_preset.value
            msg = IO.save_preset(target_name, intro, outro)
            return (gr.update(choices=IO.preset_names(), value=target_name), "", msg)

        def _io_delete(name):
            msg = IO.delete_preset(name)
            names = IO.preset_names()
            newsel = name if name in names else (names[0] if names else None)
            p = IO.get_preset(newsel) if newsel else {"intro": "", "outro": ""}
            return (gr.update(choices=names, value=newsel),
                    p.get("intro", ""), p.get("outro", ""), msg)

        io_preset.change(_io_fill, io_preset, [io_intro, io_outro])
        io_save.click(_io_save, [io_new_name, io_intro, io_outro],
                      [io_preset, io_new_name, io_status])
        io_delete.click(_io_delete, io_preset,
                        [io_preset, io_intro, io_outro, io_status])

        # U-2: control buttons must stay responsive DURING a running dub -> no queue limit
        pause_btn.click(do_live_pause, [proj, target], dashboard, concurrency_limit=None)
        resume_btn.click(do_live_resume, [proj, target], dashboard, concurrency_limit=None)
        cancel_btn.click(do_live_cancel, [proj, target], dashboard, concurrency_limit=None)

        def do_dash_tick(p, t, on):
            if not on or not p:
                return gr.update()
            pipe = _get_pipe(p, t)
            if pipe is None:
                return {"state": "no active live-render for this project"}
            return pipe.snapshot()

        # BUG FIX: dashboard auto-refreshes every 2s while a live-render runs
        dash_timer.tick(do_dash_tick, [proj, target, dash_auto], dashboard,
                        concurrency_limit=None)

        def do_persistent_dub_dashboard(p, t):
            if not p:
                return {"state": "select a project"}
            job = _dub_jobs.get(f"{safe_name(p)}::{t}")
            return job or {"state": "no active/recent dub for this project and target"}

        dub_dashboard_timer.tick(do_persistent_dub_dashboard, [proj, target], dub_dashboard,
                                 concurrency_limit=None)
        dub_btn.click(do_dub,
                      [proj, target, model, script_box, voice_mode, ref_voice,
                       ref_transcript, energy, emotion_tags, live_on, group_size,
                       clear_after, instances, force_regen, denoise_on, denoise_strength,
                       fa2_on, vox_batch, io_enabled, io_intro, io_outro, narrator_speed,
                       adv_exaggeration, adv_cfg, adv_temperature, adv_steps],
                      progress)

        def do_cancel_dub(p, t):
            if not p:
                return "Pick a project."
            job_id = f"{safe_name(p)}::{t}"
            found = get_router().cancel_job(job_id)
            if found:
                return ("🛑 Cancel requested — the dub will stop after the current "
                        "cue(s) finish. Completed cues are kept; press Dub to resume.")
            return "No active dub for this project/target to cancel."

        # concurrency_limit=None so Cancel fires even while the dub occupies the queue
        cancel_dub_btn.click(do_cancel_dub, [proj, target], progress,
                             concurrency_limit=None)
    return {}


# =====================================================================
# TAB 4 — Direct Text to Audio
# =====================================================================
def build_tab4():
    with gr.Tab("4 · Direct Text to Audio"):
        gr.Markdown("### Standalone audio — no video/transcript/timeline needed")
        name = gr.Textbox(label="Direct audio project name", value="direct")
        text = gr.Textbox(label="Text to speak", lines=5)
        with gr.Row():
            target = gr.Dropdown(H.target_choices(), value="hinglish_devanagari",
                                 label="Language / script")
            model = gr.Dropdown(H.model_choices(), value="indicf5", label="Model")
            style = gr.Dropdown(["natural", "expressive", "calm"],
                                value="expressive", label="Energy")
        with gr.Row():
            voice_mode = gr.Radio(["Built-in", "Saved reference"], value="Built-in",
                                  label="Voice")
            ref_voice = gr.Dropdown([p.name for p in VOICES.glob("*.wav")],
                                    label="Reference voice")
            ref_text = gr.Textbox(label="Reference transcript (IndicF5/VoxCPM2)")
        tags = gr.Textbox(
            label="Emotion / style hint (optional)",
            placeholder="Fish: [excited] · VoxCPM2: (energetic) · Qwen3-TTS: Very happy.")
        d_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                            label="🗣 Narrator speed (1.0 = normal · pitch kept natural)")
        with gr.Accordion("AI adapt this text first (optional, uses credits)", open=False):
            adapt_ai = gr.Checkbox(False, label="Adapt Direct Text with AI before speaking")
            with gr.Row():
                adapt_provider = gr.Dropdown(providers.PROVIDERS, value="gemini",
                                             label="Adapt provider")
                adapt_model = gr.Textbox(label="Adapt model (blank = default)")
        go = gr.Button("Generate", variant="primary")
        out_wav = gr.Audio(label="WAV output")
        out_mp3 = gr.File(label="MP3 download")
        adapted_box = gr.Textbox(label="Adapted text used (if AI adapt on)", lines=3)
        status = gr.Markdown()

        def do_go(n, txt, t, mid, st, vmode, refv, reftext, tg, use_ai, ap, am, spd):
            reference = str(VOICES / refv) if (vmode == "Saved reference" and refv) else None
            # auto-fill reference transcript from sidecar if user left it blank
            if reference and not (reftext or "").strip():
                from ..directaudio import voices as _V
                reftext = _V.transcript_for_voice(refv) or reftext
            r = synth_direct(n, txt, t, mid, st, reference, reftext or None, tg or None,
                             adapt_ai=bool(use_ai), adapt_provider=ap, adapt_model=am,
                             narrator_speed=float(spd))
            if not r.get("ok"):
                return None, None, "", f"❌ {r.get('error')}"
            return (r["wav"], r.get("mp3"), r.get("final_text", "") if use_ai else "",
                    f"✅ {r.get('seconds', 0):.1f}s generated{r.get('adapted_note','')}.")

        go.click(do_go, [name, text, target, model, style, voice_mode, ref_voice,
                         ref_text, tags, adapt_ai, adapt_provider, adapt_model, d_speed],
                 [out_wav, out_mp3, adapted_box, status])
    return {}


# =====================================================================
# TAB 5 — Subtitles & Export
# =====================================================================
def build_tab5():
    cfg = C.load_config()
    ex = cfg.get("export", {})
    with gr.Tab("5 · Subtitles & Export"):
        gr.Markdown("### Export Studio")
        with gr.Row():
            proj = gr.Dropdown(H.list_projects(), label="Project")
            refresh_projects = gr.Button("↻ Projects")
            target = gr.Dropdown(H.target_choices(), value="hinglish_devanagari",
                                 label="Target edition")
            preset = gr.Dropdown(ex.get("presets", []), value="YouTube Standard",
                                 label="Export preset")
        summary = gr.Markdown("Current Export Summary will appear here.")
        with gr.Accordion("Timing & Video Length", open=True):
            timing = gr.Dropdown(ex.get("timing_modes", []),
                                 value=ex.get("default_timing"), label="Timing mode")
            with gr.Row():
                compress = gr.Checkbox(False, label="Compress long gaps")
                compress_ms = gr.Number(1000, label="Compress gaps > (ms)")
                keep_ms = gr.Number(300, label="Keep after (ms)")
        with gr.Accordion("Add My Target Subtitles", open=False):
            captions_on = gr.Checkbox(False, label="Add target captions (burn my English/Hindi text)")
            caption_in_mask = gr.Checkbox(
                False, label="Place captions where the Chinese subs were (in the masked area)")
            caption_src = gr.Radio(
                ["Derive from narration (no credits)", "Generate AI captions", "Import SRT"],
                value="Derive from narration (no credits)", label="Caption source")
            import_srt = gr.File(label="Custom SRT")
            caption_preview_btn = gr.Button("Preview captions (first 10 lines)")
            caption_preview = gr.Textbox(label="Caption preview (retimed to final timeline)",
                                         lines=8)
        with gr.Accordion("Hide Original Chinese Burned-In Subtitles", open=False):
            mask_on = gr.Checkbox(False, label="Hide Chinese subtitles (applies to whole video)")
            with gr.Row():
                mask_type = gr.Dropdown(MASK_TYPES, value="Blur + dark band", label="Mask type")
                mask_color = gr.Dropdown(MASK_COLORS, value="black",
                                         label="Band / cover color")
            with gr.Row():
                mx = gr.Slider(0, 3840, value=308, step=2, label="X")
                my = gr.Slider(0, 2160, value=946, step=2, label="Y")
            with gr.Row():
                mw = gr.Slider(2, 3840, value=854, step=2, label="Width")
                mh = gr.Slider(2, 2160, value=90, step=2, label="Height")
            with gr.Row():
                strength = gr.Slider(1, 40, value=10, step=1, label="Blur/pixelate strength")
                opacity = gr.Slider(0, 1, value=0.6, step=0.05, label="Band opacity")
                preview_time = gr.Slider(0, 600, value=5, step=1, label="Preview time (s)")
            with gr.Row():
                preview_box_only = gr.Checkbox(
                    False, label="Show yellow box only (where the mask applies)")
                preview_btn = gr.Button("👁 Preview mask on a real frame")
            mask_preview = gr.Image(label="Mask preview (real frame from your video)")
            gr.Markdown("_Move the sliders and click Preview to see the yellow box "
                        "(area) or the actual blur/cover on a real frame. The mask is "
                        "burned across the ENTIRE video on export._")
        with gr.Accordion("Audio & BGM", open=False):
            audio_mode = gr.Radio(["Clean Dub"], value="Clean Dub", label="Audio mode")
            bgm_on = gr.Checkbox(False, label="Add BGM (music bed, auto-ducked under narration)")
            with gr.Row():
                bgm_file = gr.Dropdown([p.name for p in BGM.glob("*")],
                                       label="BGM file (from data/bgm/)")
                bgm_refresh = gr.Button("↻")
            with gr.Row():
                bgm_upload = gr.Audio(label="…or upload a BGM track",
                                      type="filepath", sources=["upload"])
                bgm_save_name = gr.Textbox(label="Save as", value="my_bgm")
                bgm_save_btn = gr.Button("💾 Save BGM")
            bgm_status = gr.Markdown()
            with gr.Row():
                duck = gr.Checkbox(True, label="Sidechain ducking")
                loudnorm = gr.Checkbox(True, label="Loudness normalize (-16 LUFS)")
                limiter = gr.Checkbox(True, label="True peak limiter (-1.5 dBTP)")
            with gr.Row():
                bgm_gain = gr.Slider(-30, 0, value=-12, step=1,
                                     label="BGM level (dB) — lower = quieter music bed")
                duck_depth = gr.Slider(1, 20, value=8, step=1,
                                       label="Duck depth — higher = music dips more under narration")
        with gr.Accordion("YouTube Title, Description & Tags", open=False):
            meta_on = gr.Checkbox(False, label="Generate metadata")
            meta_lang = gr.Dropdown(ex.get("metadata_languages", []),
                                    value="Hinglish Devanagari Preferred", label="Metadata language")
            meta_provider = gr.Dropdown(providers.PROVIDERS, value="gemini", label="Provider")
            gen_meta = gr.Button("Generate metadata")
            meta_out = gr.JSON(label="Metadata")
            meta_limits = gr.Markdown()
            with gr.Row():
                dl_txt = gr.Button("TXT"); dl_json = gr.Button("JSON"); dl_csv = gr.Button("CSV")
            meta_file = gr.File(label="Download metadata")
        with gr.Accordion("Output Files & Quality Report", open=True):
            quick_export = gr.Button("Quick Export MP4", variant="primary")
            full_export = gr.Button("Full Export (with filters)", variant="primary")
            final_video = gr.Video(label="Final video")
            with gr.Row():
                final_mp4 = gr.File(label="Final MP4")
                final_srt = gr.File(label="Final SRT")
                final_script = gr.File(label="Final script")
                quality = gr.File(label="Quality JSON")
        status = gr.Markdown()

        def do_export(p, t, timing_mode, comp, comp_ms, k_ms, mask, mtype, x, y, w, h,
                      strength_v, opacity_v, caps, capsrc, full,
                      bgm_on_v, bgm_file_v, duck_v, loudnorm_v, bgm_gain_v, duck_depth_v,
                      mask_color_v="black", cap_in_mask=False):
            if not p:
                return None, None, None, None, None, "Pick a project."
            pid = safe_name(p)
            from ..export import exporter, timeline as TL, srt as SRT
            _srcv = find_source_video(pid)
            vids = [_srcv] if _srcv else []
            if not vids:
                return None, None, None, None, None, "No source video."
            cue_dir = edition_dir(pid, t) / "tts_cues"
            clean_cues = sorted(cue_dir.glob("cue_*[0-9].wav"))
            if not clean_cues:
                return None, None, None, None, None, "No dubbed cues — run Tab 3 Dub first."
            # Build Cue list from transcript timing + cleaned durations
            tr = json.loads((project_dir(pid) / "transcript" / "transcript.json")
                            .read_text(encoding="utf-8"))
            import soundfile as sf
            cues = []
            for i, seg in enumerate(tr[:len(clean_cues)]):
                info = sf.info(str(clean_cues[i]))
                cues.append(TL.Cue(idx=i, src_start=seg["start"], src_end=seg["end"],
                                   audio_seconds=info.frames / info.samplerate,
                                   text=seg["text"]))
            tline = TL.build_timeline(timing_mode, cues,
                                      compress_gaps_ms=int(comp_ms) if comp else None,
                                      keep_after_ms=int(k_ms) if comp else None)
            outdir = edition_dir(pid, t) / "exports" / "V1"
            outdir.mkdir(parents=True, exist_ok=True)
            work = outdir / "work"
            # concat cue audio into one master track
            import numpy as np
            master = []
            sr = 48000
            for seg in tline.segments:
                if seg.kind == "gap":
                    master.append(np.zeros(int(seg.out_duration * sr), dtype="float32"))
                else:
                    a, s2 = sf.read(str(clean_cues[seg.cue_idx]))
                    master.append(np.asarray(a, dtype="float32"))
            audio_master = outdir / "audio_master.wav"
            sf.write(str(audio_master), np.concatenate(master) if master else np.zeros(1), sr)
            # video segments — FAST CLEAN EXPORT: reuse cached live-render groups when
            # no video filtergraph (no mask) is selected AND cached cue IDs match.
            silent_video = outdir / "video_silent.mp4"
            used_cache = False
            if not mask:
                man = edition_dir(pid, t) / "live_render_groups" / "manifest.json"
                if man.exists():
                    import json as _json
                    groups = _json.loads(man.read_text(encoding="utf-8")).get("groups", {})
                    from ..dubbing.live_render import cache_matches_timeline
                    if cache_matches_timeline(groups, tline, timing_mode):
                        lst = work / "cached_concat.txt"; work.mkdir(parents=True, exist_ok=True)
                        lst.write_text("\n".join(
                            f"file '{groups[gk]['file']}'"
                            for gk in sorted(groups, key=lambda k: int(k))),
                            encoding="utf-8")
                        exporter.concat_video(lst, silent_video, reencode=True)
                        used_cache = True
            if not used_cache:
                lst = exporter.build_segments_concat(str(vids[0]), tline, work,
                                                     fast_copy=False)
                # re-encode final join -> clean continuous timeline (no mid-video freeze)
                exporter.concat_video(lst, silent_video, reencode=True)
            # optional mask
            video_stage = silent_video
            if mask:
                from ..export.subtitle_mask import build_mask_filter
                fc = build_mask_filter(mtype, x, y, w, h, int(strength_v),
                                       float(opacity_v), color=mask_color_v)
                masked = outdir / "video_masked.mp4"
                exporter.apply_filtergraph(video_stage, fc, masked)
                video_stage = masked
            final = outdir / "final.mp4"
            if bgm_on_v and bgm_file_v:
                from ..export.bgm import bgm_mix_filter
                bgm_path = BGM / bgm_file_v
                fc = bgm_mix_filter(duck=bool(duck_v), bgm_gain_db=float(bgm_gain_v),
                                    duck_ratio=float(duck_depth_v))
                exporter.mux_audio_with_bgm(video_stage, audio_master, bgm_path,
                                            final, fc)
            elif loudnorm_v:
                from ..export.bgm import clean_dub_audio_filter
                exporter.mux_audio(video_stage, audio_master, final,
                                   audio_filter=clean_dub_audio_filter())
            else:
                exporter.mux_audio(video_stage, audio_master, final, audio_filter=None)
            # captions
            srt_path = outdir / "final.srt"
            if caps:
                texts = {c.idx: c.text for c in cues}
                subs = SRT.retime_from_timeline(texts, tline)
                SRT.write_srt(subs, str(srt_path))
                if "Import" not in capsrc and full:
                    burned = outdir / "final_subbed.mp4"
                    # If asked, position captions where the Chinese subs were (mask area)
                    fstyle = None
                    if cap_in_mask and mask:
                        fstyle = exporter.caption_style_for_mask(int(y), int(h))
                    exporter.burn_subtitles(final, srt_path, burned, force_style=fstyle)
                    final = burned
            # script + quality
            script_path = outdir / "final_script.txt"
            script_path.write_text("\n".join(c.text for c in cues), encoding="utf-8")
            q = outdir / "quality.json"
            meas = exporter.measure_loudness(str(final))
            exporter.write_quality_report(q, {
                "cues": len(cues), "timing_mode": timing_mode,
                "total_seconds": round(tline.total_seconds, 2),
                "nvenc": exporter.has_nvenc(),
                "loudness": meas,
                "loudness_verdict": exporter.loudness_verdict(meas)})
            return (str(final), str(final), str(srt_path) if srt_path.exists() else None,
                    str(script_path), str(q), _export_done_msg(pid, final.name))

        def do_mask_preview(p, t, mtype, x, y, w, h, strength_v, opacity_v, ptime,
                            color_v, box_only):
            if not p:
                return None
            import subprocess
            pid = safe_name(p)
            _srcv = find_source_video(pid)
            if not _srcv:
                return None
            from ..export.subtitle_mask import build_mask_filter, build_preview_rect
            outdir = edition_dir(pid, t) / "exports" / "V1"; outdir.mkdir(parents=True, exist_ok=True)
            frame = outdir / "mask_preview.png"
            # box_only: draw the YELLOW rectangle so you can see WHERE it applies.
            # otherwise: render the ACTUAL blur/cover effect on a real frame (WYSIWYG).
            if box_only:
                fc = build_preview_rect(x, y, w, h)
            else:
                fc = build_mask_filter(mtype, x, y, w, h, int(strength_v),
                                       float(opacity_v), color=color_v)
            try:
                subprocess.run(["ffmpeg", "-y", "-ss", str(ptime), "-i", str(_srcv),
                                "-filter_complex", fc, "-map", "[v]", "-frames:v", "1",
                                str(frame)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                return str(frame)
            except Exception as e:  # noqa: BLE001
                from ..common.logging_util import get_logger
                get_logger("ui.mask").warning("mask preview failed: %s", e)
                return None

        def do_caption_preview(p, t, timing_mode):
            if not p:
                return "Pick a project."
            pid = safe_name(p)
            import json as _json
            from ..export import timeline as TL, srt as SRT
            import soundfile as sf
            trj = project_dir(pid) / "transcript" / "transcript.json"
            cue_dir = edition_dir(pid, t) / "tts_cues"
            clean = sorted(cue_dir.glob("cue_*[0-9].wav"))
            if not trj.exists() or not clean:
                return "Need a transcript (Tab 1) and dubbed cues (Tab 3) first."
            tr = _json.loads(trj.read_text(encoding="utf-8"))
            cues = []
            for i, seg in enumerate(tr[:len(clean)]):
                info = sf.info(str(clean[i]))
                cues.append(TL.Cue(idx=i, src_start=seg["start"], src_end=seg["end"],
                                   audio_seconds=info.frames / info.samplerate,
                                   text=seg["text"]))
            tl = TL.build_timeline(timing_mode, cues)
            subs = SRT.retime_from_timeline({c.idx: c.text for c in cues}, tl)
            def fmt(s):
                m, ss = divmod(int(s), 60)
                return f"{m:02d}:{ss:02d}"
            return "\n".join(f"[{fmt(s.start)}→{fmt(s.end)}] {s.text}" for s in subs[:10]) \
                or "(no captions — dub cues first)"

        def do_gen_meta(p, t, lang, prov):
            if not p:
                return {}, None, ""
            pid = safe_name(p)
            script = edition_dir(pid, t) / "exports" / "V1" / "final_script.txt"
            base = script.read_text(encoding="utf-8") if script.exists() else ""
            if not base:
                # fall back to forwarded package
                pkg = PKG.load_package(pid, t)
                base = "\n".join(pkg.get("narration_lines", [])) if pkg else ""
            prompt = META.build_metadata_prompt(base, lang, "Engaging YouTube Hinglish")
            r = providers.adapt(prov, "", "You are a YouTube metadata expert.", prompt,
                                want_json=True)
            if not r.get("ok"):
                return {"error": r.get("error")}, None, ""
            md = META.parse_metadata_json(r["text"])   # already clamped to YT limits
            outdir = edition_dir(pid, t) / "exports" / "V1"; outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "metadata.json").write_text(META.to_json(md), encoding="utf-8")
            rep = META.limits_report(md)
            report = ("**Within YouTube limits ✅** — "
                      f"title {rep['title_chars']} · description {rep['description_bytes']} · "
                      f"tags {rep['tags_chars']} · hashtags {rep['hashtags']}")
            return md, str(outdir / "metadata.json"), report

        def do_meta_dl(p, t, md, fmt):
            if not p or not md:
                return None
            pid = safe_name(p)
            outdir = edition_dir(pid, t) / "exports" / "V1"; outdir.mkdir(parents=True, exist_ok=True)
            if fmt == "txt":
                f = outdir / "metadata.txt"; f.write_text(META.to_txt(md), encoding="utf-8")
            elif fmt == "csv":
                f = outdir / "metadata.csv"; f.write_text(META.to_csv(md), encoding="utf-8")
            else:
                f = outdir / "metadata.json"; f.write_text(META.to_json(md), encoding="utf-8")
            return str(f)

        def do_bgm_save(upath, name):
            if not upath:
                return gr.update(), "Upload a BGM file first."
            import shutil as _sh
            from pathlib import Path as _P
            BGM.mkdir(parents=True, exist_ok=True)
            ext = _P(upath).suffix or ".mp3"
            safe = safe_name(name or "my_bgm")
            dst = BGM / f"{safe}{ext}"
            _sh.copy(upath, dst)
            return (gr.update(choices=[p.name for p in BGM.glob("*")], value=dst.name),
                    f"✅ Saved BGM as **{dst.name}**. It's selected — enable 'Add BGM'.")

        def do_bgm_refresh():
            return gr.update(choices=[p.name for p in BGM.glob("*")])

        def do_refresh_projects(current):
            choices = H.list_projects()
            value = current if current in choices else (choices[-1] if len(choices) == 1 else None)
            return gr.update(choices=choices, value=value)

        refresh_projects.click(do_refresh_projects, proj, proj)
        bgm_save_btn.click(do_bgm_save, [bgm_upload, bgm_save_name], [bgm_file, bgm_status])
        bgm_refresh.click(do_bgm_refresh, outputs=bgm_file)
        preview_btn.click(do_mask_preview,
                          [proj, target, mask_type, mx, my, mw, mh, strength, opacity,
                           preview_time, mask_color, preview_box_only], mask_preview)
        caption_preview_btn.click(do_caption_preview, [proj, target, timing],
                                  caption_preview)
        gen_meta.click(do_gen_meta, [proj, target, meta_lang, meta_provider],
                       [meta_out, meta_file, meta_limits])
        dl_txt.click(lambda p, t, md: do_meta_dl(p, t, md, "txt"),
                     [proj, target, meta_out], meta_file)
        dl_json.click(lambda p, t, md: do_meta_dl(p, t, md, "json"),
                      [proj, target, meta_out], meta_file)
        dl_csv.click(lambda p, t, md: do_meta_dl(p, t, md, "csv"),
                     [proj, target, meta_out], meta_file)
        quick_export.click(
            do_export,
            [proj, target, timing, compress, compress_ms, keep_ms, mask_on, mask_type,
             mx, my, mw, mh, strength, opacity, captions_on, caption_src, gr.State(False),
             bgm_on, bgm_file, duck, loudnorm, bgm_gain, duck_depth, mask_color,
             caption_in_mask],
            [final_video, final_mp4, final_srt, final_script, quality, status])
        full_export.click(
            do_export,
            [proj, target, timing, compress, compress_ms, keep_ms, mask_on, mask_type,
             mx, my, mw, mh, strength, opacity, captions_on, caption_src, gr.State(True),
             bgm_on, bgm_file, duck, loudnorm, bgm_gain, duck_depth, mask_color,
             caption_in_mask],
            [final_video, final_mp4, final_srt, final_script, quality, status])
    return {}


# =====================================================================
# TAB 6 — Settings, Prompts, Keys & Cleanup
# =====================================================================
def build_tab6():
    with gr.Tab("6 · Settings, Prompts, Keys & Cleanup"):
        with gr.Accordion("Prompt Studio", open=False):
            gr.Markdown("Core rules, styles, global default, templates.")
            with gr.Row():
                style_name = gr.Textbox(label="Custom style name")
                style_text = gr.Textbox(label="Style description")
            with gr.Row():
                save_style = gr.Button("Save custom style")
                del_style = gr.Button("Delete custom style")
            gdefault = gr.Textbox(label="Global Default Prompt", value=prompts.get_global_default())
            save_gd = gr.Button("Save global default")
            prompt_status = gr.Markdown()
        with gr.Accordion("Saved AI Provider Keys + HuggingFace token", open=True):
            gr.Markdown("Keys stored locally (no login). HF token needed for IndicF5 & Fish.")
            with gr.Row():
                gem = gr.Textbox(label="Gemini key", type="password")
                grq = gr.Textbox(label="Groq key", type="password")
            with gr.Row():
                orr = gr.Textbox(label="OpenRouter key", type="password")
                cer = gr.Textbox(label="Cerebras key", type="password")
            save_keys = gr.Button("Save provider keys")
            hf = gr.Textbox(label=f"HuggingFace token ({HFT.token_status()})", type="password")
            save_hf = gr.Button("Save HF token")
            keys_status = gr.Markdown()
        with gr.Accordion("90-minute Keepalive", open=False):
            gr.Markdown("Keeps the GPU active. Max 90 minutes. Uses GPU machine time.")
            with gr.Row():
                start_ka = gr.Button("Start Keepalive")
                stop_ka = gr.Button("Stop Keepalive")
                ka_status = gr.Markdown("Keepalive: stopped")
        with gr.Accordion("Storage & Cleanup (10 GB budget)", open=True):
            disk_status = gr.Markdown()
            refresh_disk2 = gr.Button("↻ Show disk / installed models")
            with gr.Row():
                unload_all_models_btn = gr.Button("⏏ Unload ALL models from VRAM", variant="secondary")
                vram_unload_status = gr.Markdown(
                    "Unloads TTS workers and Whisper from GPU RAM; it does **not** delete model files.")
            gr.Markdown("### 🕒 Session Mode (temporary big-model use)")
            gr.Markdown(
                "_Lightning bills PERSISTENT storage above 10 GB (daily). The running "
                "Studio disk is part of the machine you already pay for. So you can "
                "TEMPORARILY use bigger models (even Fish ~15 GB) during a session, then "
                "**Cleanup for exit** to drop back under 10 GB before you stop the Studio "
                "→ no storage charge. Auto-cleanup on exit is armed while Session Mode is ON._")
            with gr.Row():
                session_toggle = gr.Checkbox(
                    True, label="Session Mode (ON by default — models like VoxCPM2 "
                    "just work; auto-cleanup on exit keeps persistent disk under budget)")
                footprint_btn = gr.Button("Show persistent footprint")
            with gr.Row():
                keep_whisper_exit = gr.Checkbox(True, label="Keep Whisper on exit (~3 GB, reused)")
                cleanup_exit_btn = gr.Button("🧹 Cleanup for exit (drop under 10 GB)",
                                             variant="primary")
            session_status = gr.Markdown()
            gr.Markdown("---")
            gr.Markdown("**Evict a TTS model** (delete its venv + weights to free space). "
                        "_Whisper is kept cached and never disk-evicted (only unloads from VRAM)._")
            with gr.Row():
                evict_model_pick = gr.Dropdown(
                    ["chatterbox", "indicf5", "voxcpm2", "qwen3tts", "vibevoice", "fish"],
                    label="TTS model to evict")
                evict_btn = gr.Button("Evict selected model", variant="stop")
            evict_all_btn = gr.Button("Evict ALL models (keep only app)", variant="stop")
            gr.Markdown("---")
            targets = gr.CheckboxGroup(
                ["Projects", "Input files", "Output files", "Reference voices",
                 "BGM", "Model cache"], label="Other cleanup targets")
            gr.Markdown("_⚠ 'Projects' = your transcripts, dubbed cues & exports. "
                        "It is NEVER auto-deleted — only if you tick the box below AND click clean._")
            confirm_projects = gr.Checkbox(
                False, label="Yes, delete my project data (transcripts/cues/exports)")
            cleanup_btn = gr.Button("Clean selected", variant="stop")
            cleanup_status = gr.Markdown()

        def do_disk_status():
            from ..common.diskmanager import (disk_free_gb, installed_model_venvs,
                                              MODEL_PEAK_GB, WHISPER_RESIDENT_GB, PROTECTED)
            inst = installed_model_venvs()
            eff = 10 - WHISPER_RESIDENT_GB
            lines = [f"**Disk free:** {disk_free_gb():.1f} GB / 10 GB budget",
                     f"**Installed:** {', '.join(inst) or 'none (frugal)'}",
                     f"**Whisper:** kept cached permanently (~{WHISPER_RESIDENT_GB:.0f} GB), "
                     f"never disk-evicted — only leaves VRAM.",
                     f"**Budget for one TTS model:** ~{eff:.0f} GB (after Whisper reserve)",
                     "", "Approx peak disk per TTS model (one at a time):"]
            for m, gb in MODEL_PEAK_GB.items():
                fit = "✅ fits" if gb <= eff else "❌ too big (use API/bigger disk)"
                lines.append(f"- {m}: ~{gb} GB {fit}")
            return "\n".join(lines)

        def do_unload_all_models():
            """Free GPU RAM only; unlike eviction this preserves installed models."""
            errors = []
            try:
                get_router().unload_all()
            except Exception as e:  # noqa: BLE001
                errors.append(f"TTS: {e}")
            try:
                from ..transcribe import whisper_engine as WE
                WE.release_gpu(reason="user requested unload all models")
            except Exception as e:  # noqa: BLE001
                errors.append(f"Whisper: {e}")
            from ..dubbing.vram_manager import free_vram_gb
            free = free_vram_gb()
            suffix = f" GPU free: {free:.1f} GB." if free is not None else " GPU status unavailable."
            if errors:
                return "⚠ Unload attempted; " + " · ".join(errors) + suffix
            return "✅ All loaded TTS models and Whisper were unloaded from VRAM." + suffix

        def do_evict(mid):
            from ..common.diskmanager import evict_model, disk_free_gb
            freed = evict_model(mid)
            return f"Evicted {mid}: freed {freed:.1f} GB. Disk free: {disk_free_gb():.1f} GB."

        def do_evict_all():
            from ..common.diskmanager import evict_all_except, disk_free_gb
            freed = evict_all_except(None)
            return f"Evicted all models: freed {freed:.1f} GB. Disk free: {disk_free_gb():.1f} GB."

        def do_session_toggle(on):
            from ..common import diskmanager as D
            D.set_session_mode(bool(on))
            if on:
                return ("✅ Session Mode ON — bigger models allowed temporarily. "
                        "Auto-cleanup on exit is armed. Run 'Cleanup for exit' (or just "
                        "stop the Studio) to drop under 10 GB.")
            return "Session Mode OFF — normal 10 GB persistent budget enforced."

        def do_footprint():
            from ..common.diskmanager import persistent_footprint_gb
            fp = persistent_footprint_gb()
            lines = [f"**Persistent footprint (survives Studio stop):** "
                     f"{fp['total_gb']:.1f} GB  ·  free disk: {fp['free_gb']:.1f} GB",
                     "", "Breakdown:"]
            for k, v in sorted(fp["parts"].items(), key=lambda x: -x[1]):
                if v > 0.01:
                    lines.append(f"- {k}: {v:.1f} GB")
            over = fp["total_gb"] - 10
            lines.append("")
            lines.append("✅ Under 10 GB — no storage charge." if fp["total_gb"] < 10
                         else f"⚠ {over:.1f} GB over 10 GB — run 'Cleanup for exit' before stopping.")
            return "\n".join(lines)

        def do_cleanup_exit(keep_w):
            from ..common.diskmanager import cleanup_for_exit
            r = cleanup_for_exit(keep_whisper=bool(keep_w))
            return r["message"]

        def do_save_keys(g, gq, o, c):
            msgs = []
            for prov, val in [("gemini", g), ("groq", gq), ("openrouter", o), ("cerebras", c)]:
                if val:
                    msgs.append(K.save_key(prov, val))
            return " ".join(msgs) or "No keys changed."

        def do_save_hf(tok):
            msg = HFT.save_hf_token(tok)
            return msg

        def do_cleanup(sel, confirm_projects):
            import shutil
            from ..common import paths as P
            # SAFETY: Projects (transcripts, dubbed cues, exports) are irreplaceable.
            # They can ONLY be deleted if the user ALSO ticks the explicit confirm box.
            if "Projects" in sel and not confirm_projects:
                return ("⚠ 'Projects' selected but NOT confirmed. Your transcripts / dubbed "
                        "cues / exports were NOT deleted. Tick 'Yes, delete my project data' "
                        "to confirm, or unselect Projects.")
            mp = {"Projects": P.PROJECTS, "Input files": P.INPUT, "Output files": P.OUTPUT,
                  "Reference voices": P.VOICES, "BGM": P.BGM, "Model cache": P.HF_CACHE}
            n = 0
            for s in sel:
                d = mp.get(s)
                if d and d.exists():
                    for item in d.iterdir():
                        try:
                            shutil.rmtree(item) if item.is_dir() else item.unlink(); n += 1
                        except Exception:
                            pass
            return f"Cleaned {n} items from: {', '.join(sel)}"

        save_style.click(lambda n, t: prompts.save_custom_style(n, t),
                         [style_name, style_text], prompt_status)
        del_style.click(lambda n: prompts.delete_custom_style(n), style_name, prompt_status)
        save_gd.click(lambda t: prompts.set_global_default(t), gdefault, prompt_status)
        save_keys.click(do_save_keys, [gem, grq, orr, cer], keys_status)
        save_hf.click(do_save_hf, hf, keys_status)
        from ..common import keepalive as KA
        start_ka.click(lambda: KA.start(), outputs=ka_status)
        stop_ka.click(lambda: KA.stop(), outputs=ka_status)
        refresh_disk2.click(do_disk_status, outputs=disk_status)
        unload_all_models_btn.click(do_unload_all_models, outputs=vram_unload_status,
                                    concurrency_limit=None)
        session_toggle.change(do_session_toggle, session_toggle, session_status)
        footprint_btn.click(do_footprint, outputs=session_status)
        cleanup_exit_btn.click(do_cleanup_exit, keep_whisper_exit, session_status)
        evict_btn.click(do_evict, evict_model_pick, cleanup_status)
        evict_all_btn.click(do_evict_all, outputs=cleanup_status)
        cleanup_btn.click(do_cleanup, [targets, confirm_projects], cleanup_status)
    return {}


# =====================================================================
# TAB 7 — Live Logs (model download / venv setup / transcribe / dub / export)
# =====================================================================
def build_logs_tab():
    from ..common import logging_util as LOG
    with gr.Tab("7 · Live Logs"):
        gr.Markdown(
            "### Live activity log\n"
            "Shows real progress from every stage — **model install & weight "
            "download**, venv setup, transcription, dubbing (cue by cue), and export. "
            "Auto-refreshes every 3s while this tab is open; use the buttons to refresh "
            "now, clear the view, or download the full log.")
        with gr.Row():
            auto_refresh = gr.Checkbox(True, label="Auto-refresh (every 3s)")
            max_lines = gr.Slider(100, 2000, value=400, step=100, label="Lines to show")
            refresh_now = gr.Button("↻ Refresh now", variant="primary")
            download_log = gr.Button("⬇ Download full log")
        log_box = gr.Textbox(label="studio.log (newest at bottom)", lines=26,
                             value=LOG.tail_log(400), autoscroll=True, max_lines=26)
        log_file = gr.File(label="Full log file", visible=False)
        timer = gr.Timer(3.0)

        def _tail(n):
            return LOG.tail_log(int(n))

        def _maybe_tail(on, n):
            # only re-read when auto-refresh is enabled (avoids needless work)
            return LOG.tail_log(int(n)) if on else gr.update()

        def _download():
            p = LOG.log_path()
            if not p.exists():
                return gr.update(visible=False)
            return gr.update(value=str(p), visible=True)

        refresh_now.click(_tail, max_lines, log_box)
        timer.tick(_maybe_tail, [auto_refresh, max_lines], log_box)
        download_log.click(_download, outputs=log_file)
    return {}
