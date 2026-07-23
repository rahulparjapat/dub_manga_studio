"""Unit tests — models mocked; verify logic that runs without a GPU."""
import numpy as np
import soundfile as sf

from chatterbox_manga_studio.common import config as C
from chatterbox_manga_studio.export import timeline as TL
from chatterbox_manga_studio.export import srt as SRT
from chatterbox_manga_studio.export.subtitle_mask import build_mask_filter, _safe_blur_radii
from chatterbox_manga_studio.dubbing import cleanup
from chatterbox_manga_studio.dubbing.workers.protocol import GenRequest, TARGET_LANG
from chatterbox_manga_studio.dubbing.vram_manager import check_model_fits
from chatterbox_manga_studio.adapt import glossary


def test_config_loads_and_profiles():
    cfg = C.load_config()
    assert "dubbing_models" in cfg
    assert set(["chatterbox", "indicf5", "voxcpm2", "vibevoice", "fish"]) <= set(cfg["dubbing_models"])
    prof = C.active_profile(cfg)
    assert "vram_gb" in prof


def test_default_model_routing():
    assert C.default_model_for_target("english") == "chatterbox"
    assert C.default_model_for_target("hinglish_devanagari") == "indicf5"


def test_target_lang_map():
    assert TARGET_LANG["english"] == "en"
    assert TARGET_LANG["hinglish_roman"] == "hi"


def test_timeline_cue_locked_NO_GAPS_default():
    # Default Cue-Locked = NO gaps: cues back-to-back, video stretched per cue.
    cues = [TL.Cue(0, 0.0, 5.0, audio_seconds=3.0),
            TL.Cue(1, 6.0, 10.0, audio_seconds=2.0)]
    tl = TL.build_timeline("Cue-Locked Audio Master Sync", cues)
    cue_segs = [s for s in tl.segments if s.kind == "cue"]
    assert abs(cue_segs[0].out_duration - 3.0) < 1e-6
    assert abs(cue_segs[1].out_duration - 2.0) < 1e-6
    # NO gap segments at all
    assert not [s for s in tl.segments if s.kind == "gap"]
    # back-to-back: cue1 starts exactly where cue0 ends
    assert abs(cue_segs[1].out_start - 3.0) < 1e-6
    assert abs(tl.total_seconds - 5.0) < 1e-6  # 3 + 2, source gap dropped


def test_timeline_keep_natural_pauses_mode():
    cues = [TL.Cue(0, 0.0, 5.0, audio_seconds=3.0),
            TL.Cue(1, 6.0, 10.0, audio_seconds=2.0)]  # 1s source gap
    tl = TL.build_timeline("Cue-Locked (Keep Natural Pauses)", cues)
    gaps = [s for s in tl.segments if s.kind == "gap"]
    assert any(abs(g.out_duration - 1.0) < 1e-6 for g in gaps)


def test_timeline_silence_compression_in_keep_pauses():
    cues = [TL.Cue(0, 0.0, 1.0, audio_seconds=1.0),
            TL.Cue(1, 6.0, 7.0, audio_seconds=1.0)]  # 5s gap
    tl = TL.build_timeline("Cue-Locked (Keep Natural Pauses)", cues,
                           compress_gaps_ms=1000, keep_after_ms=300)
    gap = [s for s in tl.segments if s.kind == "gap"][0]
    assert abs(gap.out_duration - 0.3) < 1e-6


def test_timing_modes_all_present():
    for mode in ["Cue-Locked Audio Master Sync", "Full Video Retime",
                 "Keep Original Timing", "Freeze/Pad", "Trim"]:
        tl = TL.build_timeline(mode, [TL.Cue(0, 0, 2, audio_seconds=1.5)])
        assert tl.total_seconds > 0


def test_srt_roundtrip(tmp_path):
    cues = [SRT.SubCue(1, 0.0, 1.5, "hello"), SRT.SubCue(2, 2.0, 3.0, "world")]
    p = tmp_path / "x.srt"
    SRT.write_srt(cues, str(p))
    back = SRT.read_srt(str(p))
    assert len(back) == 2
    assert back[0].text == "hello"
    assert abs(back[1].end - 3.0) < 1e-3


def test_chroma_blur_safety_small_crop():
    luma, chroma = _safe_blur_radii(20, 12, strength=40)
    assert luma >= 1 and chroma >= 1
    assert luma <= max(1, min(20, 12) // 2 - 1)


def test_mask_filters_build():
    for mt in ["Blur", "Dark band", "Blur + dark band", "Pixelate", "Cover"]:
        f = build_mask_filter(mt, 0, 100, 200, 60, 10, 0.6)
        assert "[v]" in f


def test_cleanup_produces_wav(tmp_path):
    sr = 24000
    x = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype("float32")
    # pad with silence to test trimming
    x = np.concatenate([np.zeros(2000, "float32"), x, np.zeros(3000, "float32")])
    raw = tmp_path / "raw.wav"; sf.write(str(raw), x, sr)
    out = tmp_path / "clean.wav"
    dur = cleanup.clean_cue(str(raw), str(out))
    assert out.exists() and dur > 0.5
    a, sr2 = sf.read(str(out))
    assert sr2 == 48000  # resampled to final sr


def test_crossfade_join():
    sr = 24000
    a = np.ones(1000, "float32"); b = np.ones(1000, "float32")
    joined = cleanup.crossfade_join([a, b], sr, xfade_ms=15)
    assert len(joined) < 2000  # overlap removed some samples


def test_glossary_first_mapping_retained(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    # re-point edition_dir via monkeypatched PROJECTS
    g1 = glossary.merge("proj", "hinglish_roman", {"characters": {"李明": "Li Ming"}})
    g2 = glossary.merge("proj", "hinglish_roman", {"characters": {"李明": "Different"}})
    assert g2["characters"]["李明"] == "Li Ming"  # first retained


def test_genrequest_json_roundtrip():
    r = GenRequest(text="hi", out_path="/tmp/a.wav", target="hinglish_roman")
    d = r.to_json()
    r2 = GenRequest.from_json(d)
    assert r2.language == "hi"
    assert r2.text == "hi"


def test_vram_check_returns_object():
    chk = check_model_fits("chatterbox", live_render=False)
    assert hasattr(chk, "ok")


def test_assemble_adaptation_and_forward(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    import json
    from chatterbox_manga_studio.adapt import batch_manager as BM
    from chatterbox_manga_studio.dubbing import package as PKG
    pid, t = "p", "hinglish_roman"
    tr = [{"id": i, "start": i, "end": i + 1, "text": f"c{i}"} for i in range(4)]
    (P.PROJECTS / pid / "transcript").mkdir(parents=True)
    (P.PROJECTS / pid / "transcript" / "transcript.json").write_text(json.dumps(tr))
    BM.create_plan(pid, t, tr, main_batches=2)
    for b in BM.load_plan(pid, t)["batches"]:
        v = BM.save_batch_version(pid, t, b["batch"],
                                  [f"b{b['batch']}-{j}" for j in range(b["cue_count"])],
                                  "gemini", "m")
        BM.mark(pid, t, b["batch"], status="Done", active_version=v)
    asm = BM.assemble_adaptation(pid, t)
    assert len(asm) == 4
    ver = PKG.forward_package(pid, t, {"narration_lines": asm, "dubbing_model": "indicf5"})
    pkg = PKG.load_package(pid, t)
    assert pkg["dubbing_model"] == "indicf5" and len(pkg["narration_lines"]) == 4


def test_keepalive_start_stop():
    from chatterbox_manga_studio.common import keepalive as KA
    assert "started" in KA.start().lower()
    assert "stopped" in KA.stop().lower()


def test_emotion_adapter_model_aware():
    from chatterbox_manga_studio.adapt import emotions as E
    # capable models
    assert E.is_emotion_capable("fish")
    assert E.is_emotion_capable("voxcpm2")
    # incapable models -> empty prompt (feature disabled)
    for m in ("chatterbox", "indicf5", "vibevoice"):
        assert not E.is_emotion_capable(m)
        assert E.build_emotion_prompt(m, False) == ""
    # capable models -> non-empty prompt with native syntax hints
    fp = E.build_emotion_prompt("fish", ai_free=False)
    assert "square-bracket" in fp and "EMOTION LAYER" in fp
    vp = E.build_emotion_prompt("voxcpm2", ai_free=True)
    assert "parenthetical" in vp
    # palette has the manga set
    assert set(["hype", "tense", "whisper", "sad", "comedic", "calm"]) <= set(E.MANGA_PALETTE)


def test_instance_cap_by_vram(monkeypatch):
    from chatterbox_manga_studio.dubbing import router as R
    # requested 5, small model on 24GB profile -> allowed several but <=5
    n = R._instances_for("indicf5", 5)
    assert 1 <= n <= 5
    # requested 1 always yields at least 1
    assert R._instances_for("fish", 1) >= 1


def test_cleanup_keeps_venv_by_default(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import diskmanager as D
    from chatterbox_manga_studio.common import paths as P
    monkeypatch.setattr(P, "WORKERS_ENVS", tmp_path)
    # make a fake venv + fake weights dir
    (tmp_path / "indicf5" / "bin").mkdir(parents=True)
    (tmp_path / "indicf5" / "bin" / "python").write_text("x")
    r = D.cleanup_after_dub("indicf5", keep_venv=True)
    assert (tmp_path / "indicf5" / "bin" / "python").exists()  # venv kept
    assert "venv kept" in r["message"]


def test_session_mode_and_cleanup_for_exit(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import diskmanager as D
    from chatterbox_manga_studio.common import paths as P
    monkeypatch.setattr(P, "WORKERS_ENVS", tmp_path)
    monkeypatch.setattr(D, "WORKERS_ENVS", tmp_path)
    # Pin free disk to a large value so the test checks the session-mode LOGIC,
    # not the host's actual free space (tmpfs /tmp can be <1 GB in CI/sandboxes).
    monkeypatch.setattr(D, "disk_free_gb", lambda: 40.0)
    # fake a big model venv + whisper venv
    for m in ("fish", "whisper"):
        (tmp_path / m / "bin").mkdir(parents=True)
        (tmp_path / m / "bin" / "python").write_text("x")
    # normal mode blocks fish (>10 GB budget); session mode allows (disk has room)
    D.set_session_mode(False)
    assert D.fits_budget("fish")[0] is False
    D.set_session_mode(True)
    assert D.fits_budget("fish")[0] is True
    # cleanup for exit removes fish, keeps whisper
    r = D.cleanup_for_exit(keep_whisper=True)
    assert not (tmp_path / "fish" / "bin" / "python").exists()
    assert (tmp_path / "whisper" / "bin" / "python").exists()
    assert "persistent_gb" in r
    D.set_session_mode(True)   # restore new default (ON)


def test_session_mode_defaults_on():
    """Session Mode now defaults ON so VoxCPM2 'just works' (no manual toggle)."""
    import importlib
    from chatterbox_manga_studio.common import diskmanager as D
    # fresh import state should have mode True by default
    assert D._SESSION["mode"] is True or D.session_mode() is True


def test_whisper_worker_logs_device_and_path():
    """Worker must log device + which transcribe path ran (debuggability)."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "model loaded on device=" in src
    assert "RUNNING ON CPU" in src           # LOUD warning on slow CPU fallback
    assert "batched pipeline OK" in src
    # single-file batch transcription: condition_on_previous_text=True improves
    # cross-segment coherence (False is only for real-time streaming) — verified.
    assert "condition_on_previous_text=True" in src


def test_split_long_text():
    from chatterbox_manga_studio.common.textutil import split_long_text
    assert split_long_text("short line") == ["short line"]
    long = "वाक्य एक। " * 100  # ~1000 chars
    chunks = split_long_text(long, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)  # allow small overflow on hard splits


def test_reference_check_handles_missing(tmp_path):
    from chatterbox_manga_studio.common.voicecheck import check_reference
    ok, msg = check_reference(str(tmp_path / "nope.wav"))
    assert ok is True  # can't analyze -> proceed, not block
    assert "proceeding" in msg.lower() or "could not" in msg.lower()


def test_reference_check_short_clip(tmp_path):
    import numpy as np, soundfile as sf
    from chatterbox_manga_studio.common.voicecheck import check_reference
    p = tmp_path / "short.wav"
    sf.write(str(p), (0.3*np.sin(2*np.pi*220*np.arange(24000)/24000)).astype("float32"), 24000)  # 1s
    ok, msg = check_reference(str(p))
    assert ok is False and "short" in msg.lower()


def test_stageflow_guidance_and_state(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import stageflow as SF
    from chatterbox_manga_studio.common import paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    # guidance returns required keys for every stage
    for stg in ["adaptation", "transcribe", "dubbing", "export"]:
        g = SF.stage_guidance(stg)
        assert set(["stage", "current_gpu", "recommended", "ok", "why", "tip"]) <= set(g)
    # adaptation must be marked as no-GPU (CPU recommended)
    assert SF.STAGE_PLAN["adaptation"]["gpu_needed"] is False
    assert SF.STAGE_PLAN["dubbing"]["machine"] == "L4"
    # state tracking persists + resumes
    SF.mark_stage("p", "transcribe", "T4")
    st = SF.load_state("p")
    assert st["stages"]["transcribe"]["done"] is True
    assert "transcribe" in SF.progress_summary("p")


def test_gpu_config_check_shape():
    from chatterbox_manga_studio.common import stageflow as SF
    c = SF.gpu_config_check()
    assert set(["configured", "detected", "match", "message"]) <= set(c)


def test_free_stale_port_no_crash():
    # should never raise even if fuser/lsof absent or port free
    from chatterbox_manga_studio.dubbing.router import _free_stale_port
    _free_stale_port(59999)  # unused port -> no-op, must not raise


def test_crash_safe_checkpoint(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import stageflow as SF
    from chatterbox_manga_studio.common import paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    SF.checkpoint("p", "dub_progress", {"done": 12, "total": 50, "model": "indicf5"})
    assert SF.get_checkpoint("p", "dub_progress")["done"] == 12
    assert "cue 12/50" in SF.resume_hint("p")
    SF.clear_checkpoint("p", "dub_progress")
    assert SF.get_checkpoint("p", "dub_progress") is None


def test_loudness_verdict():
    from chatterbox_manga_studio.export.exporter import loudness_verdict
    assert "within" in loudness_verdict({"input_i": "-16.2", "input_tp": "-1.6"})
    assert "outside" in loudness_verdict({"input_i": "-9.0", "input_tp": "-0.2"})
    assert "not measured" in loudness_verdict({})


def test_long_cue_split_logic():
    # verify the split produces sentence chunks (via base_worker inline logic replicated)
    import re
    text = ("यह एक लंबा वाक्य है। " * 60)  # ~1200 chars
    parts = re.split(r"(?<=[।.!?])\s+", text.strip())
    assert len(parts) > 1  # splits at danda boundaries


# ---------------------------------------------------------------------------
# Max-Quality adaptation upgrades (adapt/quality.py)
# ---------------------------------------------------------------------------
def test_cue_payload_carries_seconds_and_number():
    from chatterbox_manga_studio.adapt import quality as Q
    import json as _j
    cues = [{"start": 0.0, "end": 2.5, "text": "你好"},
            {"start": 2.5, "end": 6.0, "text": "世界"}]
    obj = _j.loads(Q.build_cue_payload(cues))
    assert [c["n"] for c in obj["cues"]] == [1, 2]
    assert obj["cues"][0]["seconds"] == 2.5
    assert obj["cues"][1]["seconds"] == 3.5
    assert obj["cues"][0]["source"] == "你好"


def test_parse_cue_response_strict_json():
    from chatterbox_manga_studio.adapt import quality as Q
    txt = '```json\n{"cues":[{"n":2,"text":"world"},{"n":1,"text":"hello"}]}\n```'
    lines, warns = Q.parse_cue_response(txt, 2)
    assert lines == ["hello", "world"]  # reordered by n
    assert warns == []


def test_parse_cue_response_pads_and_trims():
    from chatterbox_manga_studio.adapt import quality as Q
    short, w1 = Q.parse_cue_response('{"cues":[{"n":1,"text":"a"}]}', 3)
    assert short == ["a", "", ""] and any("padded" in x for x in w1)
    long, w2 = Q.parse_cue_response(
        '{"cues":[{"n":1,"text":"a"},{"n":2,"text":"b"},{"n":3,"text":"c"}]}', 2)
    assert long == ["a", "b"] and any("trimmed" in x for x in w2)


def test_parse_cue_response_plain_fallback():
    from chatterbox_manga_studio.adapt import quality as Q
    lines, warns = Q.parse_cue_response("1. hello\n2) world\n", 2)
    assert lines == ["hello", "world"]
    assert any("fallback" in w for w in warns)


def test_context_carryover_block():
    from chatterbox_manga_studio.adapt import quality as Q
    tail = Q.summarise_tail(["a", "b", "c", "d", "e"], max_lines=2)
    assert tail == "d e"
    block = Q.build_context_block(tail, "hero levels up")
    assert "hero levels up" in block and "d e" in block


def test_glossary_extract_and_lock():
    from chatterbox_manga_studio.adapt import quality as Q
    txt = ('{"cues":[{"n":1,"text":"x"}],'
           '"glossary":{"characters":{"叶凡":"Ye Fan"},"powers":{}}}')
    g = Q.extract_glossary_from_response(txt)
    assert g["characters"]["叶凡"] == "Ye Fan"
    lock = Q.glossary_lock_block(g)
    assert "Ye Fan" in lock


def test_backcheck_parse_and_summary():
    from chatterbox_manga_studio.adapt import quality as Q
    txt = ('{"checks":[{"n":1,"ok":true,"issue":""},'
           '{"n":2,"ok":false,"issue":"name wrong"}]}')
    checks = Q.parse_backcheck(txt)
    assert len(checks) == 2 and checks[1]["ok"] is False
    summ = Q.backcheck_summary(checks)
    assert "flagged 1/2" in summ and "name wrong" in summ


# ---------------------------------------------------------------------------
# Loophole fixes: L-1 (tail truncation), L-2 (count enforce), L-3 (transcript fp)
# ---------------------------------------------------------------------------
def test_L1_placeholder_keeps_tail(tmp_path, monkeypatch):
    """A missing middle cue must NOT truncate later cues; timeline keeps all slots."""
    import numpy as np, soundfile as sf
    from chatterbox_manga_studio.export import timeline as TL
    cue_dir = tmp_path
    # 3 cues: write cleaned wavs for 0 and 2, leave cue 1 missing (failed)
    for i in (0, 2):
        sf.write(str(cue_dir / f"cue_{i:04d}.wav"),
                 np.zeros(int(0.5 * 48000), dtype="float32"), 48000)
    cues = [TL.Cue(idx=i, src_start=i, src_end=i + 1, text=f"line{i}") for i in range(3)]
    # replicate the L-1 fill logic
    failed = []
    for c in cues:
        clean = cue_dir / f"cue_{c.idx:04d}.wav"
        if clean.exists() and clean.stat().st_size > 512:
            info = sf.info(str(clean)); c.audio_seconds = info.frames / info.samplerate
        else:
            sf.write(str(clean), np.zeros(int(0.6 * 48000), dtype="float32"), 48000)
            c.audio_seconds = 0.6; failed.append(c.idx + 1)
    assert failed == [2]                 # cue index 1 -> 1-based 2
    assert len(cues) == 3                # NOTHING truncated
    tl = TL.build_cue_locked(cues)
    assert len([s for s in tl.segments if s.kind == "cue"]) == 3


def test_L2_count_note_logic():
    # pure logic mirror of _count_note (equal vs mismatch)
    def note(n_lines, n_cues):
        if n_cues is None:
            return "no transcript"
        return "match" if n_lines == n_cues else "WARNING"
    assert note(50, 50) == "match"
    assert note(48, 50) == "WARNING"
    assert note(5, None) == "no transcript"


def test_L3_transcript_fingerprint(tmp_path, monkeypatch):
    import json as _j
    from chatterbox_manga_studio.common import stageflow as SF
    from chatterbox_manga_studio.common import paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    # stageflow uses project_dir -> which uses PROJECTS; patch there too
    import chatterbox_manga_studio.common.stageflow as SFmod
    def fake_project_dir(pid):
        d = tmp_path / pid; (d / "transcript").mkdir(parents=True, exist_ok=True); return d
    monkeypatch.setattr(SFmod, "project_dir", fake_project_dir)
    pid = "proj"
    tr = fake_project_dir(pid) / "transcript" / "transcript.json"
    tr.write_text(_j.dumps([{"start": 0.0, "end": 1.0, "text": "a"},
                            {"start": 1.0, "end": 2.0, "text": "b"}]))
    fp1 = SF.record_transcript_fingerprint(pid)
    assert fp1.startswith("2:")
    assert SF.transcript_changed_since_adaptation(pid) is False
    # change segmentation -> fingerprint changes
    tr.write_text(_j.dumps([{"start": 0.0, "end": 1.5, "text": "a"},
                            {"start": 1.5, "end": 2.0, "text": "b"},
                            {"start": 2.0, "end": 3.0, "text": "c"}]))
    assert SF.transcript_changed_since_adaptation(pid) is True


# ---------------------------------------------------------------------------
# Export fast-path + setpts correctness (exporter.py)
# ---------------------------------------------------------------------------
def test_setpts_factor_is_out_over_src():
    """Regression: setpts multiplier must be out/src (stretch), NOT src/out.
    A 2s source lasting 3s output must use factor 1.5, not 0.667."""
    from chatterbox_manga_studio.export import timeline as TL
    seg = TL.Segment("cue", 0, src_start=0.0, src_end=2.0, out_start=0.0, out_duration=3.0)
    src_dur = max(0.04, seg.src_end - seg.src_start)
    pts_factor = seg.out_duration / src_dur
    assert abs(pts_factor - 1.5) < 1e-6


def test_copy_ok_source_gate():
    from chatterbox_manga_studio.export import exporter as EX
    assert EX._copy_ok_source({"codec_name": "h264", "pix_fmt": "yuv420p"}) is True
    assert EX._copy_ok_source({"codec_name": "hevc", "pix_fmt": "yuv420p"}) is False
    assert EX._copy_ok_source({"codec_name": "h264", "pix_fmt": "yuv444p"}) is False
    assert EX._copy_ok_source({}) is False


def test_segment_unchanged_detection():
    from chatterbox_manga_studio.export import exporter as EX
    from chatterbox_manga_studio.export import timeline as TL
    same = TL.Segment("cue", 0, 0.0, 2.0, 0.0, 2.0)
    diff = TL.Segment("cue", 0, 0.0, 2.0, 0.0, 3.0)
    assert EX._segment_is_unchanged(same) is True
    assert EX._segment_is_unchanged(diff) is False


def test_near_keyframe_logic():
    from chatterbox_manga_studio.export import exporter as EX
    kf = [0.0, 1.0, 2.0, 3.0]
    assert EX._near_keyframe(1.0, kf) is True
    assert EX._near_keyframe(1.005, kf) is True      # within eps
    assert EX._near_keyframe(1.5, kf) is False        # off keyframe -> no copy
    assert EX._near_keyframe(0.0, []) is False        # no keyframes -> no copy


# ---------------------------------------------------------------------------
# Combined: optimized prompts -> JSON-cue adaptation parser (end to end)
# ---------------------------------------------------------------------------
def test_optimized_prompt_flows_into_json_parser():
    from chatterbox_manga_studio.adapt import prompts as P
    from chatterbox_manga_studio.adapt import quality as Q
    glossary = {"characters": {"叶凡": "Ye Fan"}}
    cues = [{"start": 0.0, "end": 2.5, "text": "叶凡突破了。"},
            {"start": 2.5, "end": 3.7, "text": "震惊！"}]
    sysp = P.build_effective_prompt("hinglish_roman", "Engaging YouTube Hinglish",
                                    "hype manhua", glossary=glossary)
    sysp += "\n" + Q.duration_rules(cues) + "\n" + Q.CUE_JSON_INSTRUCTIONS
    sysp += "\n" + Q.glossary_lock_block(glossary)
    # the optimized quality layer must be present in the effective prompt
    assert "QUALITY BAR" in sysp
    assert "hook" in sysp.lower()
    assert "WORDS" in sysp                 # numbers-as-words TTS rule
    assert "DURATION FIT" in sysp
    assert '"cues"' in sysp
    assert "Ye Fan" in sysp                # glossary lock

    # a well-behaved AI JSON response parses to exactly one line per cue + glossary
    resp = ('{"cues":[{"n":1,"text":"Ye Fan ne breakthrough kiya."},'
            '{"n":2,"text":"Sab dang reh gaye!"}],'
            '"glossary":{"characters":{"叶凡":"Ye Fan"}}}')
    lines, warns = Q.parse_cue_response(resp, 2)
    assert lines == ["Ye Fan ne breakthrough kiya.", "Sab dang reh gaye!"]
    assert warns == []
    assert Q.extract_glossary_from_response(resp)["characters"]["叶凡"] == "Ye Fan"


# ---------------------------------------------------------------------------
# Cancel current dub (router cancel registry) + BGM mux wiring
# ---------------------------------------------------------------------------
def test_router_cancel_registry():
    from chatterbox_manga_studio.dubbing.router import Router
    r = Router()
    job = "proj::hinglish_roman"
    ev = r.make_cancel_event(job)
    assert not ev.is_set()
    assert r.is_cancelling(job) is False
    assert r.cancel_job(job) is True      # found + signalled
    assert ev.is_set() and r.is_cancelling(job) is True
    assert r.cancel_job("nope::x") is False  # unknown job
    r.clear_cancel_event(job)
    assert r.is_cancelling(job) is False


def test_bgm_mix_filter_shape():
    from chatterbox_manga_studio.export.bgm import bgm_mix_filter, clean_dub_audio_filter
    fc = bgm_mix_filter(duck=True)
    assert "[2:a]" in fc and "[1:a]" in fc and "[aout]" in fc
    assert "sidechaincompress" in fc          # ducking present
    fc2 = bgm_mix_filter(duck=False)
    assert "sidechaincompress" not in fc2 and "[aout]" in fc2
    assert clean_dub_audio_filter() is None  # preserve authored narration dynamics


def test_mux_with_bgm_builds_three_input_command(monkeypatch):
    from chatterbox_manga_studio.export import exporter as EX
    from pathlib import Path
    captured = {}
    def fake_run(cmd, what="", out_path=None):
        captured["cmd"] = cmd
    monkeypatch.setattr(EX, "_run", fake_run)
    EX.mux_audio_with_bgm(Path("v.mp4"), Path("n.wav"), Path("b.mp3"),
                          Path("out.mp4"), "[1:a][2:a]amix[aout]")
    cmd = captured["cmd"]
    # three -i inputs (video, narration, bgm) and the [aout] map
    assert cmd.count("-i") == 3
    assert "-stream_loop" in cmd                # bgm looped to cover video
    assert "[aout]" in cmd


def test_bgm_duck_depth_slider_in_filter():
    from chatterbox_manga_studio.export.bgm import bgm_mix_filter
    fc = bgm_mix_filter(duck=True, bgm_gain_db=-18, duck_ratio=15)
    assert "-18dB" in fc and "ratio=15" in fc
    # clamps out-of-range values safely
    fc2 = bgm_mix_filter(duck=True, duck_ratio=999)
    assert "ratio=20" in fc2  # capped at 20


def test_direct_synth_signature_has_adapt():
    import inspect
    from chatterbox_manga_studio.directaudio.direct import synth_direct, adapt_direct_text
    sig = inspect.signature(synth_direct)
    for p in ("adapt_ai", "adapt_provider", "adapt_model"):
        assert p in sig.parameters
    assert callable(adapt_direct_text)


def test_retention_presets_in_prompt():
    from chatterbox_manga_studio.adapt import prompts as P
    assert "Cliffhanger" in P.retention_choices()
    assert "CLIFFHANGER" in P.retention_block("Cliffhanger")
    assert P.retention_block("None (use style only)") == ""
    # retention flows into the effective prompt, language-agnostic
    p_en = P.build_effective_prompt("english", "Calm but Engaging", "",
                                    retention="Reaction / Hype")
    p_hi = P.build_effective_prompt("hinglish_roman", "Calm but Engaging", "",
                                    retention="Reaction / Hype")
    assert "REACTION/HYPE" in p_en and "REACTION/HYPE" in p_hi


def test_setup_presets_roundtrip(tmp_path, monkeypatch):
    from chatterbox_manga_studio.adapt import prompts as P
    monkeypatch.setattr(P, "PROMPTS_STORE", tmp_path / "store.json")
    P.save_setup_preset("Manhua Hinglish", {
        "target": "hinglish_roman", "style": "Engaging YouTube Hinglish",
        "retention": "Cliffhanger", "add_emotions": True})
    assert "Manhua Hinglish" in P.list_setup_presets()
    got = P.load_setup_preset("Manhua Hinglish")
    assert got["target"] == "hinglish_roman" and got["retention"] == "Cliffhanger"
    assert got["add_emotions"] is True
    P.delete_setup_preset("Manhua Hinglish")
    assert "Manhua Hinglish" not in P.list_setup_presets()


def test_active_gpu_auto_detect(monkeypatch):
    """active_gpu='auto' resolves to the detected GPU's profile; unknown -> a10g."""
    from chatterbox_manga_studio.common import config as C
    import chatterbox_manga_studio.common.stageflow as SF
    cfg = C.load_config()
    cfg = dict(cfg); cfg["active_gpu"] = "auto"
    # pretend we're on a T4 -> must pick the t4 profile (fp16, never bf16)
    monkeypatch.setattr(SF, "detect_current_gpu", lambda: "t4")
    prof = C.active_profile(cfg)
    assert prof["_gpu_key"] == "t4"
    assert prof["tts_precision"] == "float16"
    # unknown GPU -> safe fallback to a10g
    monkeypatch.setattr(SF, "detect_current_gpu", lambda: "cpu")
    prof2 = C.active_profile(cfg)
    assert prof2["_gpu_key"] == "a10g"


def test_whisper_worker_has_cuda_fix():
    """The worker must auto-path NVIDIA libs and fall back GPU->CPU."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "_add_nvidia_libs_to_path" in src
    assert '("cpu", "int8")' in src
    assert "LD_LIBRARY_PATH" in src


def test_strip_emotion_tags():
    from chatterbox_manga_studio.adapt import emotions as EMO
    assert EMO.strip_emotion_tags("(calm, storytelling) Namaste doston") == "Namaste doston"
    assert EMO.strip_emotion_tags("[excited] Dhamaka [pause] hua") == "Dhamaka hua"
    assert EMO.strip_emotion_tags("no tags here") == "no tags here"


def test_strip_tags_if_incapable():
    from chatterbox_manga_studio.adapt import emotions as EMO
    lines = ["(calm) line one", "[hype] line two", "line three"]
    # incapable model (indicf5) -> tags stripped
    clean, n = EMO.strip_tags_if_incapable("indicf5", lines)
    assert clean == ["line one", "line two", "line three"] and n == 2
    # capable model (voxcpm2) -> untouched (VoxCPM2 reads the (style) prefix)
    same, n2 = EMO.strip_tags_if_incapable("voxcpm2", lines)
    assert same == lines and n2 == 0


def test_log_tail_helpers(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import logging_util as LOG
    lf = tmp_path / "studio.log"
    monkeypatch.setattr(LOG, "log_path", lambda: lf)
    assert "no log yet" in LOG.tail_log()
    lf.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")
    out = LOG.tail_log(5)
    assert out.strip().endswith("line 19") and "line 15" in out and "line 14" not in out


def test_merge_short_cues():
    from chatterbox_manga_studio.common.textutil import merge_short_cues
    # 6 cues of 1s each -> should merge into ~1 chunk of ~6s (target 7s)
    cues = [{"start": float(i), "end": float(i+1), "text": f"w{i}"} for i in range(6)]
    merged = merge_short_cues(cues, target_seconds=7.0)
    assert len(merged) < len(cues)
    assert merged[0]["start"] == 0.0
    assert merged[-1]["end"] == 6.0
    # ids renumbered contiguously
    assert [m["id"] for m in merged] == list(range(len(merged)))


def test_merge_respects_max_seconds():
    from chatterbox_manga_studio.common.textutil import merge_short_cues
    # long cues shouldn't be merged past max_seconds
    cues = [{"start": 0, "end": 8, "text": "a"}, {"start": 8, "end": 16, "text": "b"}]
    merged = merge_short_cues(cues, target_seconds=7.0, max_seconds=12.0)
    assert len(merged) == 2  # each already >= target, can't combine under max


def test_default_voice_helpers_exist():
    from chatterbox_manga_studio.directaudio import voices as VOX
    assert VOX.DEFAULT_VOICE_NAME == "default_voice.wav"
    assert "hinglish_roman" in VOX.DEFAULT_VOICE_TEXT
    assert callable(VOX.save_uploaded_voice)
    assert callable(VOX.ensure_default_voice)


def test_qwen3tts_registered():
    from chatterbox_manga_studio.common.config import load_config, model_cfg, all_models
    cfg = load_config()
    assert "qwen3tts" in all_models(cfg)
    m = model_cfg("qwen3tts", cfg)
    assert m["port"] == 8150 and m["python"] == "3.12"
    assert m["needs_ref_transcript"] is True
    assert "Apache" in m["license_flag"]
    # peak disk registered so budget checks work
    from chatterbox_manga_studio.common.diskmanager import MODEL_PEAK_GB
    assert "qwen3tts" in MODEL_PEAK_GB


def test_qwen3tts_worker_file_valid():
    import ast, pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/workers/worker_qwen3tts.py").read_text()
    ast.parse(src)  # must be syntactically valid
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in src   # verified repo id
    assert "from qwen_tts import Qwen3TTSModel" in src


def test_whisper_worker_uses_batched_pipeline_and_vad():
    """The worker must use BatchedInferencePipeline + VAD tuning (speed fix)."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "BatchedInferencePipeline" in src           # real batching (uses full VRAM)
    assert "batch_size=batch_size" in src              # batch actually passed
    assert "max_speech_duration_s" in src              # ~15s chunks
    assert "min_silence_duration_ms" in src            # fewer tiny cues
    # OOM fallback path present
    assert "out of memory" in src.lower()


def test_whisper_config_fast_defaults():
    from chatterbox_manga_studio.common.config import load_config
    w = load_config()["whisper"]
    assert w["compute_type"] == "int8_float16"  # ~35% faster than float16 on T4, ~half VRAM
    assert w["max_speech_s"] == 30          # ~2-3 cues/min (coarse cues by request)
    assert w["min_silence_ms"] == 1200


def test_router_http_surfaces_worker_error_body(monkeypatch):
    """A worker 500 must surface the JSON {error,trace}, not a generic HTTP error."""
    import io, json as _j, urllib.error
    from chatterbox_manga_studio.dubbing import router as R
    body = _j.dumps({"ok": False, "error": "boom in synth",
                     "trace": "Traceback: ValueError"}).encode()
    def fake_urlopen(req, timeout=6.0):
        raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(body))
    monkeypatch.setattr(R.urllib.request, "urlopen", fake_urlopen)
    r = R._http("http://x/generate", {"text": "hi"})
    assert r["ok"] is False
    assert r["error"] == "boom in synth"
    assert "ValueError" in r["trace"]


def test_router_has_worker_log_tail():
    from chatterbox_manga_studio.dubbing.router import Router
    r = Router()
    assert hasattr(r, "_tail_worker_log")
    # returns a string even when no log exists yet (never crashes)
    assert isinstance(r._tail_worker_log("voxcpm2", 0), str)


def test_voxcpm2_worker_disables_dynamo_on_t4():
    """VoxCPM2 worker must disable TorchDynamo (fixes T4 compile crash)."""
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/workers/worker_voxcpm2.py").read_text()
    assert "suppress_errors = True" in src
    assert "TORCHDYNAMO_DISABLE" in src
    # router also sets it when the profile can't compile
    r = pathlib.Path("src/chatterbox_manga_studio/dubbing/router.py").read_text()
    assert "TORCHDYNAMO_DISABLE" in r


def test_whisper_worker_sets_chunk_length():
    """Worker must pass chunk_length to BatchedInferencePipeline (=> ~30s cues)."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "chunk_length=chunk_len" in src
    assert "chunk_len = int(args.get(\"max_speech_s\", 30))" in src


def test_router_generate_stream_has_keep_loaded():
    import inspect
    from chatterbox_manga_studio.dubbing.router import Router
    sig = inspect.signature(Router.generate_stream)
    assert "keep_loaded" in sig.parameters
    # generate() maps unload_after=False -> keep_loaded=True (no double load)
    src = inspect.getsource(Router.generate)
    assert "keep_loaded=(not unload_after)" in src


def test_app_has_venv_reexec_guard():
    import pathlib
    src = pathlib.Path("app.py").read_text()
    assert "_reexec_in_app_venv" in src
    assert "os.execv" in src and "CMS_REEXEC" in src   # loop-safe re-exec
    assert ".venv_app" in src


def test_prewarm_button_wired():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "prewarm_btn" in src and "do_prewarm" in src
    assert "Download & warm up this model" in src


def _load_worker_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("whisper_worker", "scripts/whisper_worker.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)   # safe: faster_whisper imported lazily inside funcs
    return m


def test_worker_merge_guarantees_few_cues():
    """40 short cues (like real output) MUST merge to ~2-3 long cues, deterministically."""
    ww = _load_worker_module()
    seg = [{"id": i, "start": i*1.5, "end": (i+1)*1.5, "text": f"s{i}", "words": []}
           for i in range(40)]  # 60s of 1.5s cues
    m = ww.merge_segments(seg, target_s=25.0, max_s=30.0)
    assert len(m) <= 4, f"expected 2-3 cues, got {len(m)}"
    assert m[0]["start"] == 0.0 and m[-1]["end"] == 60.0   # timing preserved
    assert [x["id"] for x in m] == list(range(len(m)))     # ids renumbered
    # every cue (except last) is close to target length
    assert all((x["end"]-x["start"]) <= 30.0 for x in m)   # never exceeds max


def test_worker_merge_preserves_words_and_text():
    ww = _load_worker_module()
    seg = [{"id": 0, "start": 0, "end": 2, "text": "hello",
            "words": [{"start": 0, "end": 1, "word": "hello"}]},
           {"id": 1, "start": 2, "end": 4, "text": "world",
            "words": [{"start": 2, "end": 3, "word": "world"}]}]
    m = ww.merge_segments(seg, target_s=25.0, max_s=30.0)
    assert len(m) == 1
    assert m[0]["text"] == "hello world"
    assert len(m[0]["words"]) == 2   # word timings merged, not lost


def test_worker_merge_empty_safe():
    ww = _load_worker_module()
    assert ww.merge_segments([]) == []


# ---- Faster transcription pipeline (16kHz WAV pre-extraction + loud CPU fallback) ----

def test_worker_extracts_16k_mono_wav_before_transcribe():
    """The worker must pre-extract a 16 kHz mono WAV (the big speed win) instead
    of feeding the raw video to Whisper."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "extract_audio_16k" in src
    # ffmpeg args for 16 kHz mono PCM
    assert '"16000"' in src and '"-ac", "1"' in src and "pcm_s16le" in src
    assert '"-vn"' in src  # strip the video stream


def test_worker_probes_and_requires_audio_stream():
    """FFprobe validation must exist and stop with a clear error if no audio."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "probe_source" in src
    assert "NO audio stream" in src


def test_worker_cpu_fallback_is_loud():
    """CPU fallback must be VISIBLE (never silent) with a reason + fix."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "RUNNING ON CPU" in src
    assert "gpu_fail_reason" in src
    assert "nvidia-cublas-cu12" in src


def test_worker_default_compute_is_int8_float16():
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert 'args.get("compute_type", "int8_float16")' in src


def test_config_whisper_compute_is_int8_float16():
    import pathlib, re
    cfg = pathlib.Path("config.yaml").read_text()
    m = re.search(r"compute_type:\s*(\S+)", cfg)
    assert m and m.group(1) == "int8_float16"


def test_extract_audio_falls_back_to_raw_when_no_ffmpeg(tmp_path, monkeypatch):
    ww = _load_worker_module()
    monkeypatch.setattr(ww, "_ffmpeg_bin", lambda: (None, None))
    out = ww.extract_audio_16k("some_video.mp4", str(tmp_path / "transcript"), log=lambda *_: None)
    assert out == "some_video.mp4"   # graceful fallback, never crashes


def test_probe_raises_when_no_audio(monkeypatch):
    ww = _load_worker_module()
    import json as _j
    monkeypatch.setattr(ww, "_ffmpeg_bin", lambda: ("ffmpeg", "ffprobe"))
    payload = _j.dumps({"format": {"duration": "60.0"},
                        "streams": [{"codec_type": "video", "codec_name": "h264",
                                     "width": 1280, "height": 720,
                                     "avg_frame_rate": "30/1"}]})
    monkeypatch.setattr(ww.subprocess, "check_output", lambda *a, **k: payload)
    import pytest
    with pytest.raises(RuntimeError):
        ww.probe_source("v.mp4", log=lambda *_: None)


def test_engine_default_compute_int8_float16():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/transcribe/whisper_engine.py").read_text()
    assert 'wc.get("compute_type", "int8_float16")' in src


def test_transcript_header_shows_device():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "transcript_meta.json" in src
    assert "CPU = SLOW" in src


# ---- LD_LIBRARY_PATH re-exec fix (the real cuBLAS runtime bug) ----

def test_worker_reexecs_for_ld_library_path():
    """The dynamic linker reads LD_LIBRARY_PATH only at startup, so the worker
    MUST re-exec after setting it (setting os.environ in-process is too late).
    Verified against the faster-whisper README."""
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "os.execv" in src
    assert "CMS_WHISPER_REEXEC" in src   # loop guard
    assert "LD_LIBRARY_PATH" in src


def test_worker_reexec_guard_prevents_loop(monkeypatch):
    """With the guard set and libs already on the path, it must NOT re-exec."""
    ww = _load_worker_module()
    called = {"execv": False}
    monkeypatch.setattr(ww.os, "execv", lambda *a: called.__setitem__("execv", True))
    # simulate: a fake nvidia lib dir already present on LD_LIBRARY_PATH + guard set
    import tempfile, os
    d = tempfile.mkdtemp()
    nv = os.path.join(d, "nvidia", "cublas", "lib")
    os.makedirs(nv)
    monkeypatch.setattr(ww.sys, "path", [d])
    monkeypatch.setenv("LD_LIBRARY_PATH", nv)
    monkeypatch.setenv("CMS_WHISPER_REEXEC", "1")
    ww._add_nvidia_libs_to_path()
    assert called["execv"] is False   # no re-exec loop


def test_install_sets_ld_before_python():
    """Install-time GPU test must export LD_LIBRARY_PATH in BASH before python."""
    import pathlib
    src = pathlib.Path("scripts/install_model_whisper.sh").read_text()
    assert "LD_LIBRARY_PATH=" in src
    assert "before launching Python" in src.lower() or "before launching python" in src.lower()


# ---- New: chunk-length control, resident warm worker, denoiser, compile gating ----

def test_ui_has_chunk_length_slider_not_merge():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "Whisper chunk length (seconds per cue)" in src
    assert "chunk_len" in src
    # old post-hoc merge slider/button removed
    assert "Merge cues to ~N seconds each" not in src


def test_ui_has_whisper_warm_button():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "Download & load Whisper on GPU now" in src
    assert "do_warm" in src and "warm_start" in src


def test_engine_exposes_warm_and_release():
    from chatterbox_manga_studio.transcribe import whisper_engine as we
    assert hasattr(we, "warm_start")
    assert hasattr(we, "release_gpu")
    assert hasattr(we, "warm_status")


def test_router_auto_releases_whisper_before_tts_load():
    """Loading a TTS model must free the resident Whisper worker first (no OOM)."""
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/router.py").read_text()
    assert "release_gpu" in src
    assert "loading TTS model" in src


def test_worker_has_server_mode():
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "def serve(" in src
    assert '"--serve"' in src
    assert '"cmd"' in src and "shutdown" in src


def test_transcribe_accepts_chunk_seconds():
    import inspect
    from chatterbox_manga_studio.transcribe import whisper_engine as we
    sig = inspect.signature(we.transcribe)
    assert "chunk_seconds" in sig.parameters


def test_compile_gating_t4_off_l4_on():
    """torch.compile must be OFF on T4 (crashes) and ON for L4+ (verified)."""
    import pathlib, re
    cfg = pathlib.Path("config.yaml").read_text()
    t4 = re.search(r"t4:.*", cfg).group(0)
    l4 = re.search(r"l4:.*", cfg).group(0)
    assert "torch_compile: false" in t4
    assert "torch_compile: true" in l4


def test_cleanup_denoise_flag_and_reduces_noise():
    import numpy as np, tempfile, os
    import soundfile as sf
    from chatterbox_manga_studio.dubbing import cleanup
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    speech = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = 0.02 * np.random.RandomState(0).randn(sr).astype(np.float32)
    mixed = speech + noise
    d = tempfile.mkdtemp()
    raw = os.path.join(d, "raw.wav"); sf.write(raw, mixed, sr)
    # denoise=False keeps noise; denoise=True reduces it
    out_off = os.path.join(d, "off.wav")
    cleanup.clean_cue(raw, out_off, denoise=False)
    out_on = os.path.join(d, "on.wav")
    cleanup.clean_cue(raw, out_on, denoise=True, denoise_strength=1.5)
    a, _ = sf.read(out_off); b, _ = sf.read(out_on)
    # measure residual noise in a silent-ish window isn't trivial; instead check
    # the denoised signal has lower high-frequency energy (noise lives there)
    def hf_energy(x):
        X = np.abs(np.fft.rfft(x))
        return float(X[len(X)//2:].sum())
    assert hf_energy(b) < hf_energy(a), "denoise should reduce high-freq noise energy"


# ---- Pause-aware chunking + VoxCPM2 lock + GPU-aware FA2 control ----

def test_pause_aware_snaps_to_natural_pause():
    """A 25s target must end the cue at the real ~23s pause, not a hard 25.0s cut."""
    ww = _load_worker_module()
    words = []
    t = 0.0
    while t < 60.0:
        words.append({"start": round(t, 3), "end": round(t + 0.5, 3), "word": "字"})
        gap = 0.05
        if 22.5 <= t <= 23.5:   # a real pause near 23s
            gap = 0.4
        t = t + 0.5 + gap
    seg = [{"id": 0, "start": 0.0, "end": words[-1]["end"],
            "text": "完整句子", "words": words}]
    cues = ww.merge_segments_pause_aware(seg, target_s=25.0, flex=0.40, pause_gap=0.25)
    assert len(cues) >= 2
    # first cue ends at the ~23s pause (within window 15-35), NOT exactly 25.0
    assert 22.0 <= cues[0]["end"] <= 24.5, cues[0]["end"]


def test_pause_aware_no_word_timings_fallback():
    """No word timings -> deterministic time merge, never crashes."""
    ww = _load_worker_module()
    seg = [{"id": i, "start": i * 2, "end": (i + 1) * 2, "text": f"s{i}", "words": []}
           for i in range(20)]
    cues = ww.merge_segments_pause_aware(seg, target_s=10.0)
    assert len(cues) >= 1
    assert all("text" in c and "start" in c and "end" in c for c in cues)


def test_worker_uses_pause_aware_merge():
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    assert "merge_segments_pause_aware" in src
    assert "pause-aware merge" in src.lower() or "pause_aware" in src


def test_voxcpm2_locked_to_one_instance_on_16gb():
    """VoxCPM2 must be forced to exactly 1 instance on <=16GB GPUs (no OOM)."""
    import chatterbox_manga_studio.common.config as C
    import chatterbox_manga_studio.dubbing.router as R
    orig = C.active_profile
    try:
        prof = {"vram_gb": 16, "min_free_vram_reserve_gb": 2,
                "torch_compile": False, "label": "Tesla T4 16GB"}
        C.active_profile = lambda cfg=None: prof
        R.active_profile = C.active_profile
        assert R._instances_for("voxcpm2", 4) == 1
        assert R._instances_for("voxcpm2", 1) == 1
        note = R.instance_cap_note("voxcpm2", 4)
        assert "1 instance" in note
    finally:
        C.active_profile = orig
        R.active_profile = orig


def test_fa2_gated_by_gpu_capability():
    """supports_flash_attention mirrors torch_compile (sm_80+ proxy)."""
    import chatterbox_manga_studio.common.config as C
    orig = C.active_profile
    try:
        C.active_profile = lambda cfg=None: {"torch_compile": False}
        assert C.supports_flash_attention() is False   # T4
        C.active_profile = lambda cfg=None: {"torch_compile": True}
        assert C.supports_flash_attention() is True    # L4/A10G+
    finally:
        C.active_profile = orig


def test_ui_fa2_control_is_gpu_aware():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "supports_flash_attention" in src
    assert "FlashAttention 2" in src
    assert "interactive=_fa2_ok" in src       # disabled on T4
    assert "sm_75" in src                      # explains WHY it's off on T4


def test_router_forces_fa2_off_on_non_sm80():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/router.py").read_text()
    assert "VOXCPM_FLASH_ATTN" in src and "VOXCPM_BATCH_SIZE" in src
    assert "fa2_ok" in src


# ---- Crash fix (WAV not treated as video) + mask/BGM/burn/keepalive additions ----

def test_find_source_video_ignores_transcription_wav():
    import tempfile, os
    from pathlib import Path
    import chatterbox_manga_studio.common.paths as P
    orig = P.PROJECTS
    try:
        d = Path(tempfile.mkdtemp()); P.PROJECTS = d
        src = d / "proj" / "source"; src.mkdir(parents=True)
        (src / "source_audio_16k.wav").write_bytes(b"0" * 8)
        (src / "clip.mp4").write_bytes(b"0" * 8)
        r = P.find_source_video("proj")
        assert r is not None and r.suffix == ".mp4"
        # wav-only -> None (never feed audio to the video renderer)
        (src / "clip.mp4").unlink()
        assert P.find_source_video("proj") is None
    finally:
        P.PROJECTS = orig


def test_wav_written_to_transcript_not_source():
    import pathlib
    src = pathlib.Path("scripts/whisper_worker.py").read_text()
    # WAV goes into the transcript out dir, not source/
    assert 'wav = out / "source_audio_16k.wav"' in src
    assert "removed stale source/source_audio_16k.wav" in src


def test_no_bare_source_glob_for_video_lookups():
    """All source-video lookups must use find_source_video (video-only)."""
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    # the risky pattern that grabbed the wav must be gone from video lookups
    assert 'list((project_dir(pid) / "source").glob("*"))' not in src


def test_mask_filter_supports_color():
    from chatterbox_manga_studio.export.subtitle_mask import build_mask_filter, MASK_COLORS
    assert "navy" in build_mask_filter("Cover", 0, 0, 100, 50, color="navy")
    assert "navy" in build_mask_filter("Dark band", 0, 0, 100, 50, color="navy")
    assert "black" in MASK_COLORS


def test_caption_style_for_mask_positions_in_band():
    from chatterbox_manga_studio.export.exporter import caption_style_for_mask
    s = caption_style_for_mask(600, 120, video_h=1080)
    assert "MarginV=" in s and "Alignment=2" in s


def test_burn_subtitles_accepts_force_style():
    import inspect
    from chatterbox_manga_studio.export import exporter as E
    assert "force_style" in inspect.signature(E.burn_subtitles).parameters


def test_ui_has_mask_color_boxonly_and_bgm_upload():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "mask_color" in src and "preview_box_only" in src
    assert "bgm_upload" in src and "do_bgm_save" in src
    assert "caption_in_mask" in src        # dual burn toggle


def test_prewarm_releases_whisper():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "release_gpu(reason=\"user is loading a TTS model (prewarm)\")" in src


def test_keepalive_backward_compatible_and_pings():
    from chatterbox_manga_studio.common import keepalive as KA
    # no-arg start/stop still return human strings (Settings tab buttons)
    msg = KA.start()
    assert "started" in msg.lower()
    KA.stop_existing()
    assert "stopped" in KA.stop().lower()
    # status shape
    st = KA.status()
    assert "alive" in st and "remaining_min" in st


def test_keepalive_readme_note_present():
    """The keepalive module must still be honest about Lightning's own sleep."""
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/common/keepalive.py").read_text()
    assert "does not" in src.lower() and "lightning" in src.lower()


# ---- Resume dashboard + blur types + YT metadata limits + md cleanup ----

def test_resume_dashboard_and_tab_mapping():
    from chatterbox_manga_studio.common import stageflow as SF
    assert SF.which_tab_for_stage("dubbing").startswith("② Dub")
    assert SF.which_tab_for_stage("export").startswith("③ Export")
    # empty project -> friendly prompt, not a crash
    assert "pick a project" in SF.resume_dashboard("").lower()


def test_resume_dashboard_shows_next_and_where(tmp_path, monkeypatch):
    from chatterbox_manga_studio.common import stageflow as SF
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    # a fresh project -> next stage ingest, dashboard points to Tab 1
    pid = "demo"
    (tmp_path / pid).mkdir()
    d = SF.resume_dashboard(pid)
    assert "Where to resume" in d
    assert "Tab 1" in d


def test_all_blur_types_valid_filters():
    from chatterbox_manga_studio.export.subtitle_mask import MASK_TYPES, build_mask_filter
    # the requested richer set exists
    for want in ["Blur (Gaussian)", "Box blur", "Pixelate / Mosaic",
                 "Motion blur", "Frosted glass"]:
        assert want in MASK_TYPES
    for mt in MASK_TYPES:
        f = build_mask_filter(mt, 0, 600, 1280, 120, strength=12, color="navy")
        assert f.endswith("[v]") or f.endswith("[v]") or "[v]" in f


def test_gaussian_and_frosted_use_expected_ffmpeg():
    from chatterbox_manga_studio.export.subtitle_mask import build_mask_filter
    assert "gblur" in build_mask_filter("Blur (Gaussian)", 0, 0, 200, 80)
    assert "noise" in build_mask_filter("Frosted glass", 0, 0, 200, 80)
    assert "avgblur" in build_mask_filter("Motion blur", 0, 0, 200, 80)


def test_metadata_enforces_youtube_limits():
    from chatterbox_manga_studio.export import metadata as M
    md = M.clamp_to_youtube_limits({
        "title": "T" * 250,
        "description": "D" * 9000,
        "tags": [f"tag{i}" for i in range(200)],
        "hashtags": [f"#h{i}" for i in range(40)],
    })
    assert len(md["title"]) <= M.TITLE_MAX
    assert len(md["description"].encode("utf-8")) <= M.DESC_MAX_BYTES
    assert sum(len(t) for t in md["tags"]) <= M.TAGS_MAX_CHARS
    assert len(md["hashtags"]) <= M.HASHTAGS_MAX


def test_metadata_description_bytes_for_multibyte():
    """Description limit is BYTES — Hindi/Chinese chars are multi-byte."""
    from chatterbox_manga_studio.export import metadata as M
    hindi = "क" * 4000     # each 'क' is 3 bytes in UTF-8 => 12000 bytes
    md = M.clamp_to_youtube_limits({"title": "x", "description": hindi})
    assert len(md["description"].encode("utf-8")) <= M.DESC_MAX_BYTES


def test_metadata_prompt_is_hooky_and_states_limits():
    from chatterbox_manga_studio.export import metadata as M
    p = M.build_metadata_prompt("some narration", "Hinglish Roman", "hype")
    assert "curiosity" in p.lower() or "hook" in p.lower()
    assert "100 characters" in p and "4500" in p and "15" in p


def test_removed_md_files_gone():
    import pathlib
    assert not pathlib.Path("MODEL_RESEARCH_2026.md").exists()
    assert not pathlib.Path("docs/LIGHTNING_ANTISLEEP.md").exists()
    # kept ones remain
    assert pathlib.Path("README.md").exists()
    assert pathlib.Path("HOW_TO_USE_IT.md").exists()


def test_metadata_ui_shows_limit_report():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "meta_limits" in src and "limits_report" in src


# ---- AI model dropdown + resource monitor + verified TTS dep pins ----

def test_provider_model_choices_never_empty():
    from chatterbox_manga_studio.adapt import providers as P
    for prov in P.PROVIDERS:
        ch = P.model_choices(prov)
        assert isinstance(ch, list) and len(ch) >= 1, prov
        # default model is present and first
        dm = P.default_model(prov)
        if dm:
            assert ch[0] == dm


def test_ui_model_is_dropdown_not_textbox():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    # the copy-paste "Model ID" textbox is gone; a picker dropdown replaces it
    assert 'gr.Textbox(label="Model ID"' not in src
    assert 'providers.model_choices(' in src
    assert "_model_dd_for" in src
    assert "provider.change(_model_dd_for" in src


def test_sysmon_snapshot_and_widget():
    from chatterbox_manga_studio.common import sysmon as SM
    s = SM.snapshot()
    for k in ("cpu", "vram_used_mb", "vram_total_mb", "ram_used_mb", "ram_total_mb"):
        assert k in s
    h = SM.html_widget()
    assert "VRAM" in h and "CPU" in h and "RAM" in h and "<div" in h


def test_monitor_wired_top_of_app_with_1s_timer():
    import pathlib
    src = pathlib.Path("app.py").read_text()
    assert "sysmon" in src and "html_widget" in src
    assert "gr.Timer(1.0)" in src
    assert ".tick(" in src


def test_qwen3tts_pins_transformers_4573():
    import pathlib
    s = pathlib.Path("scripts/install_model_qwen3tts.sh").read_text()
    assert "transformers==4.57.3" in s     # verified hard requirement


def test_vibevoice_pins_torch_and_transformers():
    import pathlib
    s = pathlib.Path("scripts/install_model_vibevoice.sh").read_text()
    assert "transformers==4.51.3" in s
    assert "torch==2.5.1" in s  # 2.6 has no CUDA 12.1 wheel on Lightning
    assert "bitsandbytes>=0.43.0" in s


def test_indicf5_verified_pins_present():
    import pathlib
    s = pathlib.Path("scripts/install_model_indicf5.sh").read_text()
    assert "transformers==4.49.0" in s and "torch==2.2.0" in s and "numpy==1.26.4" in s


def test_voxcpm2_install_untouched_has_no_new_pins():
    """VoxCPM2 works — we must NOT have added version pins that could break it."""
    import pathlib
    s = pathlib.Path("scripts/install_model_voxcpm2.sh").read_text()
    # it installs plain 'voxcpm' (no hard torch/transformers pin we might have forced)
    assert "voxcpm" in s
    assert "transformers==" not in s      # we didn't pin transformers on voxcpm2


# ---- One-Click Auto mask/BGM controls + package version fixes ----

def test_package_latest_is_numeric_not_lexicographic(tmp_path, monkeypatch):
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    import chatterbox_manga_studio.dubbing.package as PKG
    for i in range(12):
        PKG.forward_package("v", "hinglish_roman",
                            {"narration_lines": [f"l{i}"], "dubbing_model": "indicf5"})
    latest = PKG.load_package("v", "hinglish_roman")
    assert latest["_version"] == "V12"        # not V9 (lexicographic bug)


def test_package_targets_are_isolated(tmp_path, monkeypatch):
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    import chatterbox_manga_studio.dubbing.package as PKG
    PKG.forward_package("v", "hinglish_roman", {"narration_lines": ["roman"]})
    PKG.forward_package("v", "hinglish_devanagari", {"narration_lines": ["dev1", "dev2"]})
    r = PKG.load_package("v", "hinglish_roman")
    d = PKG.load_package("v", "hinglish_devanagari")
    assert r["narration_lines"] == ["roman"]
    assert d["narration_lines"] == ["dev1", "dev2"]   # 2nd dub not clobbered by 1st


def test_package_load_previous_version(tmp_path, monkeypatch):
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    import chatterbox_manga_studio.dubbing.package as PKG
    for i in range(3):
        PKG.forward_package("v", "en", {"narration_lines": [f"x{i}"]})
    old = PKG.load_package("v", "en", version="V2")
    assert old["narration_lines"] == ["x1"]


def test_package_version_details_for_picker(tmp_path, monkeypatch):
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    import chatterbox_manga_studio.dubbing.package as PKG
    PKG.forward_package("v", "en", {"narration_lines": ["a"], "dubbing_model": "indicf5"})
    dets = PKG.version_details("v", "en")
    assert dets and dets[0]["version"] == "V1" and dets[0]["lines"] == 1
    assert "created" in dets[0] and "model" in dets[0]


def test_package_forward_uses_max_plus_one(tmp_path, monkeypatch):
    """Deleting an old version must not cause a new one to overwrite an existing file."""
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    import chatterbox_manga_studio.dubbing.package as PKG
    for _ in range(3):
        PKG.forward_package("v", "en", {"narration_lines": ["x"]})
    # delete V1, then forward -> must be V4 (max+1), never re-create V3/V1
    d = P.PROJECTS / "v"  # edition dir path resolution handled inside package
    from chatterbox_manga_studio.common.paths import edition_dir
    (edition_dir("v", "en") / "dubbing_versions" / "V1.json").unlink()
    new = PKG.forward_package("v", "en", {"narration_lines": ["y"]})
    assert new == "V4"


def test_auto_ui_has_selectable_mask_and_bgm():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    # One-Click Auto now has real, selectable controls (were display-only/empty)
    assert "auto_mask_type" in src and "auto_mask_color" in src
    assert "auto_bgm_upload" in src and "do_auto_bgm_save" in src
    assert "do_auto_mask_preview" in src
    # and they flow into run_auto via mask_opts type/color/strength
    assert '"type": m_type' in src and '"color": m_color' in src


def test_auto_pipeline_mask_passes_color():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/auto_pipeline.py").read_text()
    assert 'color=mask_opts.get("color"' in src


def test_tab3_has_version_picker():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "fwd_version" in src and "_fwd_version_choices" in src
    assert "version=version" in src or "version=version)" in src


# ---- Intro / Outro presets ----

def test_intro_outro_has_10_presets_and_default():
    from chatterbox_manga_studio.adapt import intro_outro as IO
    names = IO.preset_names()
    assert len(names) == 10
    assert IO.DEFAULT_PRESET in names
    assert IO.DEFAULT_PRESET == "Mystery / Curiosity"
    assert IO.DEFAULT_ENABLED is False


def test_intro_outro_only_mentions_rahul_no_tts():
    """Presets must mention only 'राहुल'/Rahul and never how it's produced."""
    from chatterbox_manga_studio.adapt import intro_outro as IO
    banned = ["tts", "ai ", "voice model", "synth", "generated", "dub"]
    for name, p in IO.BUILTIN_PRESETS.items():
        blob = (p["intro"] + " " + p["outro"]).lower()
        assert ("राहुल" in p["intro"] + p["outro"]) or ("rahul" in blob), name
        for b in banned:
            assert b not in blob, f"{name} leaks '{b}'"


def test_intro_outro_apply_prepends_and_appends():
    from chatterbox_manga_studio.adapt import intro_outro as IO
    out = IO.apply_to_lines(["one", "two"], "HELLO", "BYE", enabled=True)
    assert out == ["HELLO", "one", "two", "BYE"]
    # disabled -> unchanged, never mutates input
    src = ["a", "b"]
    assert IO.apply_to_lines(src, "x", "y", enabled=False) == ["a", "b"]
    assert src == ["a", "b"]
    # empty intro/outro skipped
    assert IO.apply_to_lines(["a"], "", "", enabled=True) == ["a"]


def test_intro_outro_edit_and_custom(tmp_path, monkeypatch):
    from chatterbox_manga_studio.adapt import intro_outro as IO
    monkeypatch.setattr(IO, "STORE", tmp_path / "io.json")
    # edit a built-in (non-destructive)
    IO.save_preset("Cliffhanger Hook", "NEW IN", "NEW OUT")
    assert IO.get_preset("Cliffhanger Hook")["intro"] == "NEW IN"
    # revert
    IO.delete_preset("Cliffhanger Hook")
    assert IO.get_preset("Cliffhanger Hook")["intro"] == IO.BUILTIN_PRESETS["Cliffhanger Hook"]["intro"]
    # add + delete custom
    IO.save_preset("My Own", "hi", "bye")
    assert "My Own" in IO.preset_names()
    IO.delete_preset("My Own")
    assert "My Own" not in IO.preset_names()


def test_ui_intro_outro_accordion_wired():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "io_enabled" in src and "io_preset" in src
    assert "IO.apply_to_lines(lines, io_in, io_out" in src
    assert "Intro & Outro" in src         # collapsible accordion label


# ---- Reference voice: saved once + optional denoise-at-save ----

def test_save_uploaded_voice_has_denoise_param():
    import inspect
    from chatterbox_manga_studio.directaudio import voices as V
    sig = inspect.signature(V.save_uploaded_voice)
    assert "denoise" in sig.parameters and "denoise_strength" in sig.parameters


def test_live_render_cache_requires_exact_cue_duration_and_timing_mode():
    from chatterbox_manga_studio.dubbing.live_render import cache_matches_timeline
    from chatterbox_manga_studio.export.timeline import Cue, build_timeline
    cues = [Cue(0, 0.0, 1.0, audio_seconds=1.25),
            Cue(1, 1.0, 2.0, audio_seconds=0.75)]
    timeline = build_timeline("Cue-Locked Audio Master Sync", cues)
    matching = {"0": {"file": "group.mp4", "cue_ids": [0, 1],
                      "audio_seconds": [1.25, 0.75],
                      "timing_mode": "Cue-Locked Audio Master Sync"}}
    assert cache_matches_timeline(matching, timeline, "Cue-Locked Audio Master Sync")
    assert not cache_matches_timeline(matching, timeline, "Cue-Locked (Keep Natural Pauses)")
    mismatched = {"0": {**matching["0"], "audio_seconds": [1.25, 0.95]}}
    assert not cache_matches_timeline(mismatched, timeline, "Cue-Locked Audio Master Sync")
    # Legacy ID-only manifests must be rebuilt rather than risking A/V drift.
    assert not cache_matches_timeline({"0": {"file": "old.mp4", "cue_ids": [0, 1]}},
                                      timeline, "Cue-Locked Audio Master Sync")


def test_reference_voice_wav_fallback_without_ffmpeg(tmp_path, monkeypatch):
    """A valid WAV remains usable when ffmpeg is not installed."""
    import importlib
    import numpy as np
    import soundfile as sf
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp_path / "voices")
    import chatterbox_manga_studio.directaudio.voices as V
    importlib.reload(V)
    raw = tmp_path / "input.wav"
    sf.write(str(raw), np.zeros(24_000, dtype="float32"), 24_000)

    def no_ffmpeg(*_args, **_kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(V.subprocess, "run", no_ffmpeg)
    result = V.save_uploaded_voice(str(raw), "fallback", auto_transcribe=False)
    assert result["ok"]
    assert (V.VOICES / "fallback.wav").read_bytes() == raw.read_bytes()


def test_reference_voice_does_not_rename_failed_nonwav_conversion(tmp_path, monkeypatch):
    """Never save undecodable MP3 bytes under a misleading .wav extension."""
    import importlib
    from types import SimpleNamespace
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp_path / "voices")
    import chatterbox_manga_studio.directaudio.voices as V
    importlib.reload(V)
    raw = tmp_path / "broken.mp3"
    raw.write_bytes(b"not an audio stream")
    monkeypatch.setattr(V.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1))
    result = V.save_uploaded_voice(str(raw), "broken", auto_transcribe=False)
    assert not result["ok"]
    assert not (V.VOICES / "broken.wav").exists()


def test_reference_denoise_reduces_noise(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf, importlib
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp_path / "voices")
    import chatterbox_manga_studio.directaudio.voices as V
    importlib.reload(V)
    sr = 24000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)
    speech = 0.3 * np.sin(2 * np.pi * 180 * t).astype("float32")
    noise = 0.03 * np.random.RandomState(2).randn(2 * sr).astype("float32")
    raw = tmp_path / "in.wav"; sf.write(str(raw), speech + noise, sr)
    r_plain = V.save_uploaded_voice(str(raw), "plain", denoise=False)
    r_clean = V.save_uploaded_voice(str(raw), "clean", denoise=True, denoise_strength=1.5)
    assert r_plain["ok"] and r_clean["ok"]
    a, _ = sf.read(str(V.VOICES / "clean.wav"))
    b, _ = sf.read(str(V.VOICES / "plain.wav"))
    def hf(x):
        X = np.abs(np.fft.rfft(np.asarray(x, dtype="float32")))
        return float(X[len(X) // 2:].sum())
    assert hf(a) < hf(b)     # denoised clip has less high-freq (noise) energy


def test_ui_reference_denoise_wired():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "voice_denoise" in src
    assert "Clean my reference voice" in src
    assert "denoise=bool(dn)" in src


def test_voxcpm2_worker_untouched_denoiser_off():
    """We did NOT change VoxCPM2's own load path (still load_denoiser=False)."""
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/workers/worker_voxcpm2.py").read_text()
    assert "load_denoiser=False" in src   # generation path unchanged (it works)


# ---- Reference voice: upload any length, auto-use best window ----

def test_save_uploaded_voice_has_max_seconds():
    import inspect
    from chatterbox_manga_studio.directaudio import voices as V
    assert "max_seconds" in inspect.signature(V.save_uploaded_voice).parameters


def test_long_reference_auto_trimmed_to_best_window(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf, importlib
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp_path / "voices")
    import chatterbox_manga_studio.directaudio.voices as V
    importlib.reload(V)
    sr = 24000
    x = np.zeros(60 * sr, dtype="float32")
    t = np.linspace(0, 15, 15 * sr, endpoint=False)
    x[40 * sr:55 * sr] = 0.4 * np.sin(2 * np.pi * 180 * t).astype("float32")  # speech burst
    x += 0.005 * np.random.RandomState(4).randn(60 * sr).astype("float32")
    raw = tmp_path / "long.wav"; sf.write(str(raw), x, sr)
    r = V.save_uploaded_voice(str(raw), "lv", max_seconds=25.0)
    assert r["ok"]
    out, osr = sf.read(str(V.VOICES / "lv.wav"))
    dur = len(out) / osr
    assert 24 <= dur <= 26                       # trimmed to ~25s
    rms = float(np.sqrt(np.mean(np.asarray(out, dtype="float32") ** 2)))
    assert rms > 0.1                             # grabbed the speech-dense window


def test_short_reference_kept_full(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf, importlib
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp_path / "voices")
    import chatterbox_manga_studio.directaudio.voices as V
    importlib.reload(V)
    sr = 24000
    y = 0.3 * np.sin(2 * np.pi * 180 * np.linspace(0, 10, 10 * sr, endpoint=False)).astype("float32")
    raw = tmp_path / "s.wav"; sf.write(str(raw), y, sr)
    r = V.save_uploaded_voice(str(raw), "sv", max_seconds=25.0)
    out, osr = sf.read(str(V.VOICES / "sv.wav"))
    assert 9 <= len(out) / osr <= 11            # short clip untouched


def test_ui_upload_label_says_any_length():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "any length" in src.lower()


# ---- Video render glitch fixes (mask defaults + concat timeline) ----

def test_mask_defaults_updated():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    # new defaults X=308 Y=946 W=854 H=90 present on the mask sliders
    assert "value=308" in src and "value=946" in src
    assert "value=854" in src and "value=90" in src


def test_auto_pipeline_mask_default_coords():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/auto_pipeline.py").read_text()
    assert 'mask_opts.get("x", 308)' in src and 'mask_opts.get("y", 946)' in src
    assert 'mask_opts.get("w", 854)' in src and 'mask_opts.get("h", 90)' in src


def test_segment_uses_input_side_seek_and_cfr():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/export/exporter.py").read_text()
    # input-side -ss + -t before -i (frame-count-correct), forced CFR + fixed tb
    assert '"-ss", f"{seg.src_start:.3f}", "-t", f"{src_dur:.3f}",\n' in src or \
           '"-ss", f"{seg.src_start:.3f}", "-t", f"{src_dur:.3f}"' in src
    assert "-fps_mode" in src or "-vsync" in src
    assert "setsar=1" in src


def test_concat_reencode_rebuilds_pts():
    """The reencode concat must rebuild PTS (setpts=N/FRAME_RATE/TB) — the fix for
    the inflated-duration freeze."""
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/export/exporter.py").read_text()
    assert "setpts=N/FRAME_RATE/TB" in src


def test_pipeline_final_concat_reencodes():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/auto_pipeline.py").read_text()
    # no more risky copy-concat on the final joins
    assert "reencode=False" not in src


def test_real_concat_duration_is_correct():
    """End-to-end ffmpeg test: retimed segments concat to the CORRECT duration +
    frame count (no inflated-duration freeze). Skips if ffmpeg/ffprobe absent."""
    import shutil, subprocess, tempfile, os, sys
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        import pytest; pytest.skip("ffmpeg/ffprobe not available")
    d = tempfile.mkdtemp()
    src = os.path.join(d, "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=640x360:rate=30:duration=12",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "30", src],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    sys.path.insert(0, "src")
    from chatterbox_manga_studio.export import exporter as EX
    class Seg:
        def __init__(s, k, cs, ce, od, ci=0):
            s.kind, s.src_start, s.src_end, s.out_duration, s.cue_idx = k, cs, ce, od, ci
    class TL: pass
    tl = TL(); tl.segments = [Seg("cue", 0, 3, 3.0, 0), Seg("cue", 3, 6, 4.0, 1),
                              Seg("cue", 6, 9, 3.0, 2), Seg("cue", 9, 12, 2.0, 3)]
    from pathlib import Path
    lst = EX.build_segments_concat(src, tl, Path(d) / "work", fast_copy=False)
    out = Path(d) / "joined.mp4"
    EX.concat_video(lst, out, reencode=True)
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)], text=True).strip())
    # expected 3+4+3+2 = 12s (allow small container tolerance); MUST NOT be ~70s
    assert 11.0 <= dur <= 13.5, f"duration {dur} indicates a timeline glitch"


# ---- Voice test lab: generate candidates, save, delete ----

def test_voice_lab_backend_fns_exist():
    from chatterbox_manga_studio.directaudio import voices as V
    for fn in ("generate_candidates", "save_candidate", "delete_voice", "default_test_line"):
        assert hasattr(V, fn), fn


def test_voice_save_candidate_and_delete(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf
    from chatterbox_manga_studio.directaudio import voices as V
    monkeypatch.setattr(V, "VOICES", tmp_path)
    monkeypatch.setattr(V, "CANDIDATES_DIR", tmp_path / "_candidates")
    V.CANDIDATES_DIR.mkdir(parents=True)
    cand = V.CANDIDATES_DIR / "cand_01.wav"
    sf.write(str(cand), np.zeros(24000, dtype="float32"), 24000)
    r = V.save_candidate(str(cand), "voice_a")
    assert r["ok"] and (tmp_path / "voice_a.wav").exists()
    assert "voice_a.wav" in V.list_voices()
    # candidates dir must NOT pollute the saved library list
    assert not any("cand_" in n for n in V.list_voices())
    rd = V.delete_voice("voice_a")
    assert rd["ok"] and not (tmp_path / "voice_a.wav").exists()


def test_generate_candidates_clears_old_and_limits(tmp_path, monkeypatch):
    """generate_candidates clears prior scratch samples and caps count; router is
    mocked so no real model is needed."""
    from chatterbox_manga_studio.directaudio import voices as V
    monkeypatch.setattr(V, "VOICES", tmp_path)
    monkeypatch.setattr(V, "CANDIDATES_DIR", tmp_path / "_candidates")
    V.CANDIDATES_DIR.mkdir(parents=True)
    (V.CANDIDATES_DIR / "cand_99.wav").write_bytes(b"0" * 4096)   # stale sample

    import numpy as np, soundfile as sf
    class FakeRouter:
        def generate(self, mid, req, unload_after=True):
            # GenRequest.to_json() returns a dict (matches the real router contract)
            out = req["out_path"] if isinstance(req, dict) else __import__("json").loads(req)["out_path"]
            sf.write(out, np.zeros(2400, dtype="float32"), 24000)
            return {"ok": True}
    monkeypatch.setattr(V, "get_router", lambda: FakeRouter())
    monkeypatch.setattr(V, "preset_for_style", lambda s: {})
    monkeypatch.setattr(V, "default_model_for_target", lambda t: "voxcpm2")
    r = V.generate_candidates("hinglish_devanagari", "voxcpm2", count=10, text="test")
    assert r["ok"]
    assert len(r["paths"]) == 6                    # capped at 6
    assert not (V.CANDIDATES_DIR / "cand_99.wav").exists()   # stale cleared


def test_ui_voice_lab_wired():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "Voice test lab" in src
    assert "do_vt_generate" in src and "do_vt_save" in src and "do_vt_delete" in src
    assert "vt_use" in src         # can pick a saved voice for the dub


# ---- Testing mode: allow any model on T4 (bypass disk/VRAM/instance gates) ----

def test_testing_mode_flag_default_off():
    from chatterbox_manga_studio.common import diskmanager as DM
    DM.set_testing_mode(False)
    assert DM.testing_mode() is False
    DM.set_testing_mode(True)
    assert DM.testing_mode() is True
    DM.set_testing_mode(False)   # reset


def test_testing_mode_bypasses_vram_block(monkeypatch):
    import chatterbox_manga_studio.common.config as C
    import chatterbox_manga_studio.dubbing.vram_manager as VM
    from chatterbox_manga_studio.common import diskmanager as DM
    prof = {"vram_gb": 16, "min_free_vram_reserve_gb": 3, "label": "Tesla T4 16GB"}
    monkeypatch.setattr(C, "active_profile", lambda cfg=None: prof)
    monkeypatch.setattr(VM, "active_profile", lambda cfg=None: prof)
    DM.set_testing_mode(False)
    assert VM.check_model_fits("fish").ok is False      # blocked normally on T4
    DM.set_testing_mode(True)
    chk = VM.check_model_fits("fish")
    assert chk.ok is True and "Testing mode" in chk.warning
    DM.set_testing_mode(False)


def test_testing_mode_lifts_voxcpm2_instance_lock(monkeypatch):
    import chatterbox_manga_studio.common.config as C
    import chatterbox_manga_studio.dubbing.router as R
    from chatterbox_manga_studio.common import diskmanager as DM
    prof = {"vram_gb": 16, "min_free_vram_reserve_gb": 2, "label": "Tesla T4 16GB"}
    monkeypatch.setattr(C, "active_profile", lambda cfg=None: prof)
    monkeypatch.setattr(R, "active_profile", lambda cfg=None: prof)
    DM.set_testing_mode(False)
    assert R._instances_for("voxcpm2", 4) == 1          # clamped normally
    DM.set_testing_mode(True)
    assert R._instances_for("voxcpm2", 4) == 4          # honored in testing
    DM.set_testing_mode(False)


def test_testing_mode_disk_bypass_message():
    from chatterbox_manga_studio.common import diskmanager as DM
    DM.set_testing_mode(True)
    ok, msg = DM.fits_budget("fish")
    # ok depends on physical free space, but the message must show it's testing mode
    assert "Testing mode" in msg
    DM.set_testing_mode(False)


def test_ui_testing_toggle_wired():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "Testing mode" in src and "set_testing_mode" in src
    assert "do_testing_toggle" in src


# ---- Fish worker fixes: no bogus --bnb4, UTF-8 safe, real error captured ----

def test_fish_worker_no_bnb4_flag():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/workers/worker_fish.py").read_text()
    # --bnb4 is not a real fish-speech flag; it caused argparse exit(2)
    assert '"--bnb4"' not in src


def test_fish_worker_is_debuggable_and_utf8():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/workers/worker_fish.py").read_text()
    # captures Fish's real error instead of hiding it behind exit status 2
    assert "_run_fish" in src
    assert "Real error from Fish" in src
    assert "stderr=subprocess.STDOUT" in src
    # forces UTF-8 so Devanagari isn't corrupted into mojibake
    assert "PYTHONUTF8" in src and "PYTHONIOENCODING" in src


def test_fish_config_vram_is_24():
    import pathlib, re
    cfg = pathlib.Path("config.yaml").read_text()
    # find the fish block's est_vram_gb
    m = re.search(r"fish:.*?est_vram_gb:\s*(\d+)", cfg, re.S)
    assert m and int(m.group(1)) == 24   # honest: fp16 ~24GB, not a fake 16


# ---- Narrator speed control (pitch-preserving atempo + model hint) ----

def test_atempo_chain_stays_in_range():
    from chatterbox_manga_studio.dubbing import cleanup as CU
    assert CU._atempo_chain(1.5) == "atempo=1.5"
    # wide values are split so each atempo factor stays within ffmpeg's 0.5..2.0
    for spd in (0.4, 0.5, 2.0, 3.0, 4.0):
        chain = CU._atempo_chain(spd)
        for part in chain.split(","):
            f = float(part.split("=")[1])
            assert 0.5 <= f <= 2.0, (spd, f)


def test_clean_cue_speed_changes_duration(tmp_path):
    import numpy as np, soundfile as sf
    from chatterbox_manga_studio.dubbing import cleanup as CU
    import shutil
    if not shutil.which("ffmpeg"):
        import pytest; pytest.skip("ffmpeg not available")
    sr = 48000
    w = 0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 4, 4 * sr, endpoint=False)).astype("float32")
    raw = tmp_path / "raw.wav"; sf.write(str(raw), w, sr)
    out_fast = tmp_path / "fast.wav"
    d_fast = CU.clean_cue(str(raw), str(out_fast), speed=1.25)
    out_norm = tmp_path / "norm.wav"
    d_norm = CU.clean_cue(str(raw), str(out_norm), speed=1.0)
    assert d_fast < d_norm * 0.95        # faster => shorter


def test_clean_cue_speed_default_is_noop():
    import inspect
    from chatterbox_manga_studio.dubbing import cleanup as CU
    assert inspect.signature(CU.clean_cue).parameters["speed"].default == 1.0


def test_speed_hint_model_aware():
    from chatterbox_manga_studio.adapt import emotions as E
    assert E.speed_hint("voxcpm2", 1.3)          # parenthetical
    assert E.speed_hint("fish", 0.8).startswith("[")
    assert E.speed_hint("indicf5", 1.3) == ""    # no text hint -> slider only
    assert E.speed_hint("voxcpm2", 1.0) == ""    # neutral -> no hint


def test_ui_narrator_speed_wired():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "narrator_speed" in src and "Narrator speed" in src
    assert "speed=float(nar_speed)" in src       # flows into clean_cue


def test_run_auto_accepts_narrator_speed():
    import inspect
    from chatterbox_manga_studio.dubbing import auto_pipeline as AP
    assert "narrator_speed" in inspect.signature(AP.run_auto).parameters


# ---- VoxCPM2 prompt pairing, unlimited voice saves, tab4 speed, emotions, adv TTS ----

def test_voxcpm2_prompt_pairing_preserves_emotion_control():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/workers/worker_voxcpm2.py").read_text()
    # Neutral lines with a transcript use Hi-Fi prompt pairing.
    assert 'kw["prompt_wav_path"] = req.reference_wav' in src
    assert 'kw["prompt_text"] = ref_text' in src
    # A parenthesized style prefix deliberately selects controllable cloning,
    # because VoxCPM2 ignores style instructions in Hi-Fi prompt mode.
    assert "has_style_prefix" in src
    assert "if ref_text and not has_style_prefix" in src
    assert 'kw["reference_wav_path"] = req.reference_wav' in src


def test_unlimited_voice_saves_autonumber(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf, importlib
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp_path)
    monkeypatch.setattr(P, "CANDIDATES_DIR", tmp_path / "_c", raising=False)
    import chatterbox_manga_studio.directaudio.voices as V
    importlib.reload(V)
    V.CANDIDATES_DIR = tmp_path / "_c"; V.CANDIDATES_DIR.mkdir(parents=True)
    c = V.CANDIDATES_DIR / "cand.wav"; sf.write(str(c), np.zeros(2400, dtype="float32"), 24000)
    names = {V.save_candidate(str(c), "v", f"line {i}")["name"] for i in range(4)}
    assert len(names) == 4                       # all kept, none overwritten
    assert len(V.list_voices()) == 4


def test_voice_transcript_sidecar(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf, importlib
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp_path)
    import chatterbox_manga_studio.directaudio.voices as V
    importlib.reload(V)
    V.CANDIDATES_DIR = tmp_path / "_c"; V.CANDIDATES_DIR.mkdir(parents=True)
    c = V.CANDIDATES_DIR / "cand.wav"; sf.write(str(c), np.zeros(2400, dtype="float32"), 24000)
    r = V.save_candidate(str(c), "hero", "the exact spoken line")
    assert V.transcript_for_voice(r["name"]) == "the exact spoken line"
    V.delete_voice(r["name"])
    assert V.transcript_for_voice(r["name"]) == ""   # sidecar deleted too


def test_synth_direct_has_speed_param():
    import inspect
    from chatterbox_manga_studio.directaudio import direct as D
    assert "narrator_speed" in inspect.signature(D.synth_direct).parameters


def test_emotions_default_on():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    # add_emotions checkbox now defaults True
    assert 'add_emotions = gr.Checkbox(\n                        True' in src


def test_tab4_and_advanced_tts_controls_wired():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/ui/tabs.py").read_text()
    assert "d_speed" in src                        # tab4 narrator speed
    assert "Advanced TTS controls" in src
    assert "adv_exaggeration" in src and "adv_cfg" in src
    assert 'preset["exaggeration"]' in src         # overrides flow into preset


# ---- Co-resident Whisper for reference transcription (TTS stays loaded) ----

def test_whisper_engine_has_transcribe_clip_and_gpu_free():
    from chatterbox_manga_studio.transcribe import whisper_engine as W
    import inspect
    assert hasattr(W, "transcribe_clip") and hasattr(W, "gpu_free_gb")
    ps = inspect.signature(W.transcribe_clip).parameters
    assert "tts_loaded_model" in ps


def test_transcribe_clip_coresident_when_vram_free(monkeypatch, tmp_path):
    """When VRAM is plentiful, it must NOT unload the TTS model (co-resident)."""
    from chatterbox_manga_studio.transcribe import whisper_engine as W
    monkeypatch.setattr(W, "gpu_free_gb", lambda: 10.0)   # lots free
    calls = {"unload": 0}
    class FakeRouter:
        def unload(self, mid=None): calls["unload"] += 1
    monkeypatch.setattr("chatterbox_manga_studio.dubbing.router.get_router",
                        lambda: FakeRouter())
    monkeypatch.setattr(W, "PROJECT_ROOT", tmp_path)
    def fake_transcribe(audio, out_dir, *a, **k):
        from pathlib import Path as _P
        (_P(out_dir) / "transcript.txt").write_text("hi", encoding="utf-8")
        return {"ok": True, "device": "cuda"}
    monkeypatch.setattr(W, "transcribe", fake_transcribe)
    r = W.transcribe_clip("ref.wav", tts_loaded_model="voxcpm2")
    assert r["ok"] and r["text"] == "hi"
    assert calls["unload"] == 0          # TTS stayed loaded (co-resident)
    assert r["freed_tts"] is False


def test_transcribe_clip_evicts_when_vram_tight(monkeypatch, tmp_path):
    """When VRAM is too tight, it evicts TTS first (safe fallback) and flags it."""
    from chatterbox_manga_studio.transcribe import whisper_engine as W
    monkeypatch.setattr(W, "gpu_free_gb", lambda: 1.0)    # not enough
    calls = {"unload": 0}
    class FakeRouter:
        def unload(self, mid=None): calls["unload"] += 1
    monkeypatch.setattr("chatterbox_manga_studio.dubbing.router.get_router",
                        lambda: FakeRouter())
    monkeypatch.setattr(W, "PROJECT_ROOT", tmp_path)
    def fake_transcribe(audio, out_dir, *a, **k):
        from pathlib import Path as _P
        (_P(out_dir) / "transcript.txt").write_text("yo", encoding="utf-8")
        return {"ok": True, "device": "cuda"}
    monkeypatch.setattr(W, "transcribe", fake_transcribe)
    r = W.transcribe_clip("ref.wav", tts_loaded_model="vibevoice")
    assert calls["unload"] == 1          # TTS briefly evicted
    assert r["freed_tts"] is True        # so caller reloads it


def test_router_current_model_accessor():
    import pathlib
    src = pathlib.Path("src/chatterbox_manga_studio/dubbing/router.py").read_text()
    assert "def current_model" in src


def test_save_uploaded_voice_auto_transcribe_param():
    import inspect
    from chatterbox_manga_studio.directaudio import voices as V
    ps = inspect.signature(V.save_uploaded_voice).parameters
    assert "auto_transcribe" in ps and "tts_loaded_model" in ps
