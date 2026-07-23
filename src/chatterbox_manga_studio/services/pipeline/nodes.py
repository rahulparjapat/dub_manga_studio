"""WorkflowEngine nodes wrapping existing Chatterbox Manga Studio business logic."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, cast

from ...adapt import providers as legacy_providers
from ...adapt import quality
from ...common.config import default_model_for_target, load_config
from ...common.paths import edition_dir, find_source_video, project_dir
from ...dubbing import cleanup
from ...dubbing.workers.protocol import TARGET_LANG, GenRequest
from ...export import exporter as EX
from ...export import srt as SRT
from ...export import timeline as TL
from ...ingest import upload
from ..model_manager import ModelSelectionCriteria
from ..provider_manager import ProviderRequest
from ..worker_pool import WorkerMatchCriteria
from ..workflow_engine import WorkflowContext
from .base import NodeExecutionResult, PipelineNode, PipelineNodeError


class IngestNode(PipelineNode):
    """Ingest a source video by wrapping existing upload/input/Drive helpers."""

    def __init__(self, services=None) -> None:
        super().__init__("ingest", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        data = self.merged_inputs(ctx)
        project_id = _require(data, "project_id")
        await ctx.update_progress(0.1, "ingest starting")
        await ctx.raise_if_cancelled()

        if data.get("upload_tmp_path") and data.get("filename"):
            path = await asyncio.to_thread(
                upload.store_uploaded, data["upload_tmp_path"], project_id, data["filename"]
            )
            message = "uploaded file stored"
        elif data.get("drive_url"):
            result = await asyncio.to_thread(upload.download_drive, data["drive_url"], project_id)
            if not result.get("ok"):
                raise PipelineNodeError(result.get("error") or "Drive download failed")
            path = result["path"]
            message = "downloaded from drive"
        elif data.get("auto_input"):
            result = await asyncio.to_thread(
                upload.auto_ingest_stable_input, project_id, data.get("min_stable_seconds", 6.0)
            )
            if not result.get("ok"):
                raise PipelineNodeError(result.get("message") or "auto ingest failed")
            path = result["path"]
            message = result.get("message", "auto ingested")
        else:
            source = Path(_require(data, "source_path"))
            if not source.exists():
                raise PipelineNodeError(f"source video not found: {source}")
            dst_dir = project_dir(project_id) / "source"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / source.name
            if source.resolve() != dst.resolve():
                await asyncio.to_thread(shutil.copy2, source, dst)
            path = str(dst)
            message = "source copied"

        await ctx.update_progress(1.0, "ingest complete")
        result = {"project_id": project_id, "source_video": path, "message": message}
        if self.services.storage is not None:
            await self.services.storage.set_kv(f"projects:{project_id}:source_video", path)
        return NodeExecutionResult(node=self.name, data=result)


class TranscribeNode(PipelineNode):
    """Run Whisper transcription through existing transcribe.whisper_engine."""

    def __init__(self, services=None) -> None:
        super().__init__("transcribe", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        from ...transcribe import whisper_engine

        data = self.merged_inputs(ctx)
        deps = await self.all_dependency_results(ctx)
        ingest_data = _node_data(deps.get("ingest"))
        project_id = data.get("project_id") or ingest_data.get("project_id")
        source_video = data.get("source_video") or ingest_data.get("source_video")
        if not source_video and project_id:
            found = find_source_video(project_id)
            source_video = str(found) if found else None
        if not project_id or not source_video:
            raise PipelineNodeError("TranscribeNode requires project_id and source_video")

        out_dir = data.get("transcript_dir") or str(project_dir(project_id) / "transcript")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        await ctx.update_progress(0.1, "transcription queued")
        await ctx.raise_if_cancelled()
        result = await asyncio.to_thread(
            whisper_engine.transcribe,
            source_video,
            out_dir,
            data.get("source_language", "Auto"),
            data.get("chunk_seconds"),
        )
        if not result.get("ok"):
            raise PipelineNodeError(result.get("error") or "transcription failed")
        await ctx.update_progress(1.0, "transcription complete")
        payload = {
            "project_id": project_id,
            "source_video": source_video,
            "transcript_dir": out_dir,
            "transcription": result,
        }
        transcript_json = Path(out_dir) / "transcript.json"
        if transcript_json.exists():
            payload["transcript_json"] = str(transcript_json)
            payload["transcript"] = json.loads(transcript_json.read_text(encoding="utf-8"))
        return NodeExecutionResult(node=self.name, data=payload)


class TranslationNode(PipelineNode):
    """Adapt/translate transcript text by wrapping existing provider logic."""

    def __init__(self, services=None) -> None:
        super().__init__("translation", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        data = self.merged_inputs(ctx)
        await ctx.update_progress(0.05, "translation starting")
        await ctx.raise_if_cancelled()

        if data.get("adapted_lines"):
            lines = list(data["adapted_lines"])
            return NodeExecutionResult(
                node=self.name, data={"lines": lines, "raw_text": "", "warnings": []}
            )

        cues = await _load_cues(ctx, data)
        expected = int(data.get("expected_count") or len(cues) or 0)
        system_prompt = data.get(
            "system_prompt", "Translate/adapt the cues. Return JSON with a cues array."
        )
        user_content = data.get("user_content") or quality.build_cue_payload(cues)
        provider = data.get("provider")
        model = data.get("provider_model") or data.get("model") or ""

        if self.services.providers is not None and data.get("use_provider_manager"):
            response = await self.services.providers.execute(
                ProviderRequest(
                    operation="translation",
                    payload={
                        "system_prompt": system_prompt,
                        "user_content": user_content,
                        "model": model,
                        "target": data.get("target"),
                    },
                )
            )
            raw_text = str(
                response.result.get("text")
                if isinstance(response.result, dict)
                else response.result
            )
        else:
            if not provider:
                raise PipelineNodeError("TranslationNode requires provider or adapted_lines")
            response = await asyncio.to_thread(
                cast(Any, legacy_providers.adapt),
                provider,
                model,
                system_prompt,
                user_content,
                True,
            )
            if not response.get("ok"):
                raise PipelineNodeError(response.get("error") or "translation provider failed")
            raw_text = str(response.get("text", ""))

        lines, warnings = (
            quality.parse_cue_response(raw_text, expected) if expected else ([raw_text], [])
        )
        glossary = quality.extract_glossary_from_response(raw_text)
        await ctx.update_progress(1.0, "translation complete")
        return NodeExecutionResult(
            node=self.name,
            data={"lines": lines, "raw_text": raw_text, "warnings": warnings, "glossary": glossary},
        )


class QualityNode(PipelineNode):
    """Run existing adaptation quality helpers."""

    def __init__(self, services=None) -> None:
        super().__init__("quality", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        data = self.merged_inputs(ctx)
        deps = await self.all_dependency_results(ctx)
        translation = _node_data(deps.get("translation"))
        lines = list(data.get("lines") or translation.get("lines") or [])
        cues = await _load_cues(ctx, data)
        await ctx.update_progress(0.25, "quality checks")
        await ctx.raise_if_cancelled()

        duration = quality.duration_fit(cues, lines) if cues and lines else []
        backcheck_text = data.get("backcheck_text")
        checks = quality.parse_backcheck(backcheck_text) if backcheck_text else []
        summary = quality.backcheck_summary(checks) if checks else ""
        warnings = list(translation.get("warnings") or [])

        for item in duration:
            if isinstance(item, dict):
                if not item.get("ok", True):
                    warnings.append(str(item))
            elif item:
                warnings.append(str(item))
        await ctx.update_progress(1.0, "quality complete")
        return NodeExecutionResult(
            node=self.name,
            data={
                "lines": lines,
                "duration_fit": duration,
                "backcheck": checks,
                "summary": summary,
            },
            warnings=warnings,
        )


class VoiceSelectionNode(PipelineNode):
    """Resolve model and reference voice by wrapping existing voice/config helpers."""

    def __init__(self, services=None) -> None:
        super().__init__("voice_selection", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        from ...common.voicecheck import check_reference
        from ...directaudio import voices

        data = self.merged_inputs(ctx)
        target = data.get("target", "english")
        model_id = data.get("model_id")
        warnings: list[str] = []
        await ctx.update_progress(0.2, "selecting voice/model")

        if not model_id and self.services.models is not None:
            cap = await self.services.models.recommend_model(
                ModelSelectionCriteria(
                    language=target,
                    supports_voice_clone=data.get("requires_voice_clone"),
                    supports_reference_text=data.get("requires_reference_text"),
                    max_vram=data.get("max_vram"),
                )
            )
            model_id = cap.model_id if cap else None
        if not model_id:
            model_id = default_model_for_target(target)

        reference_wav = data.get("reference_wav")
        reference_text = data.get("reference_text") or ""
        voice_name = data.get("voice_name")
        if voice_name and not reference_wav:
            candidate = Path(voice_name)
            if not candidate.is_absolute():
                from ...common.paths import VOICES

                candidate = VOICES / voice_name
            reference_wav = str(candidate)
            reference_text = reference_text or voices.transcript_for_voice(candidate.name)

        if reference_wav:
            ok, msg = await asyncio.to_thread(check_reference, reference_wav)
            if not ok:
                warnings.append(msg)
            else:
                warnings.append(msg)

        if self.services.workers is not None:
            matches = await self.services.workers.match_workers(
                WorkerMatchCriteria(model_id=model_id, language=target)
            )
            if matches:
                warnings.append(f"matched {len(matches)} worker(s) for {model_id}")
        return NodeExecutionResult(
            node=self.name,
            data={
                "target": target,
                "model_id": model_id,
                "reference_wav": reference_wav,
                "reference_text": reference_text,
            },
            warnings=warnings,
        )


class TTSNode(PipelineNode):
    """Generate raw TTS cues by wrapping existing router/model runtime logic."""

    def __init__(self, services=None) -> None:
        super().__init__("tts", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        from ...dubbing.router import get_router

        data = self.merged_inputs(ctx)
        deps = await self.all_dependency_results(ctx)
        translation = _node_data(deps.get("translation")) or await _result_for(ctx, "translation")
        voice = _node_data(deps.get("voice_selection"))
        lines = list(data.get("lines") or translation.get("lines") or [])
        if not lines:
            raise PipelineNodeError("TTSNode requires narration lines")
        project_id = (
            data.get("project_id")
            or voice.get("project_id")
            or _node_data(deps.get("ingest")).get("project_id")
        )
        target = data.get("target") or voice.get("target") or "english"
        model_id = data.get("model_id") or voice.get("model_id")
        if not project_id:
            raise PipelineNodeError("TTSNode requires project_id")
        if not model_id:
            raise PipelineNodeError("TTSNode requires model_id")
        cue_dir = edition_dir(project_id, target) / "tts_cues"
        cue_dir.mkdir(parents=True, exist_ok=True)
        preset_name = data.get("energy", "expressive")
        preset = load_config().get("tts_quality", {}).get("presets", {}).get(preset_name, {})
        reqs = []
        for i, line in enumerate(lines):
            raw = cue_dir / f"cue_{i:04d}_raw.wav"
            reqs.append(
                GenRequest(
                    text=line,
                    out_path=str(raw),
                    target=target,
                    language=TARGET_LANG.get(target, "en"),
                    reference_wav=data.get("reference_wav") or voice.get("reference_wav"),
                    reference_text=data.get("reference_text") or voice.get("reference_text"),
                    preset=preset,
                    emotion_tags=data.get("emotion_tags"),
                ).to_json()
            )

        if data.get("dry_run"):
            for i, req in enumerate(reqs):
                Path(req["out_path"]).write_bytes(b"dry-run-wav")
                await ctx.update_progress((i + 1) / len(reqs), f"dry generated cue {i + 1}")
            results = [{"ok": True, "wav_path": req["out_path"], "seconds": 0.1} for req in reqs]
        else:
            loop = asyncio.get_running_loop()

            def _on_cue(i: int, result: dict[str, Any]) -> None:
                progress = (i + 1) / max(1, len(reqs))
                asyncio.run_coroutine_threadsafe(
                    ctx.update_progress(progress, f"TTS cue {i + 1}"), loop
                )

            results = await asyncio.to_thread(
                get_router().generate_stream,
                model_id,
                reqs,
                data.get("instances", 1),
                _on_cue,
                None,
                data.get("clear_cache_after", True),
                True,
                None,
                data.get("force_regenerate", False),
                ctx.cancel_event,
                data.get("keep_loaded", False),
            )
        failures = [result for result in results if not result.get("ok")]
        if failures and not data.get("allow_partial", False):
            raise PipelineNodeError(
                f"TTS failed for {len(failures)} cue(s): {failures[0].get('error')}"
            )
        raw_cues = [req["out_path"] for req in reqs]
        return NodeExecutionResult(
            node=self.name,
            data={
                "project_id": project_id,
                "target": target,
                "model_id": model_id,
                "lines": lines,
                "raw_cues": raw_cues,
                "results": results,
            },
        )


class AudioCleanupNode(PipelineNode):
    """Clean generated cue audio through existing cleanup.clean_cue."""

    def __init__(self, services=None) -> None:
        super().__init__("audio_cleanup", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        data = self.merged_inputs(ctx)
        deps = await self.all_dependency_results(ctx)
        tts = _node_data(deps.get("tts"))
        raw_cues = list(data.get("raw_cues") or tts.get("raw_cues") or [])
        project_id = data.get("project_id") or tts.get("project_id")
        target = data.get("target") or tts.get("target") or "english"
        if not raw_cues or not project_id:
            raise PipelineNodeError("AudioCleanupNode requires project_id and raw_cues")
        cue_dir = edition_dir(project_id, target) / "tts_cues"
        clean_paths: list[str] = []
        durations: list[float] = []
        for i, raw in enumerate(raw_cues):
            await ctx.raise_if_cancelled()
            out = cue_dir / f"cue_{i:04d}.wav"
            if data.get("dry_run"):
                shutil.copyfile(raw, out)
                duration = 0.1
            else:
                duration = await asyncio.to_thread(
                    cleanup.clean_cue,
                    raw,
                    str(out),
                    data.get("denoise", False),
                    data.get("denoise_strength", 1.0),
                    data.get("speed", data.get("narrator_speed", 1.0)),
                )
            clean_paths.append(str(out))
            durations.append(float(duration))
            await ctx.update_progress((i + 1) / len(raw_cues), f"cleaned cue {i + 1}")
        return NodeExecutionResult(
            node=self.name,
            data={
                "project_id": project_id,
                "target": target,
                "clean_cues": clean_paths,
                "durations": durations,
            },
        )


class RenderNode(PipelineNode):
    """Render silent video/timeline by wrapping existing timeline/export helpers."""

    def __init__(self, services=None) -> None:
        super().__init__("render", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        data = self.merged_inputs(ctx)
        deps = await self.all_dependency_results(ctx)
        cleanup_data = _node_data(deps.get("audio_cleanup"))
        translation = _node_data(deps.get("translation")) or await _result_for(ctx, "translation")
        tts = _node_data(deps.get("tts")) or await _result_for(ctx, "tts")
        project_id = (
            data.get("project_id") or cleanup_data.get("project_id") or tts.get("project_id")
        )
        target = data.get("target") or cleanup_data.get("target") or "english"
        source_video = data.get("source_video") or _node_data(deps.get("ingest")).get(
            "source_video"
        )
        if not source_video and project_id:
            found = find_source_video(project_id)
            source_video = str(found) if found else None
        clean_cues = list(data.get("clean_cues") or cleanup_data.get("clean_cues") or [])
        durations = list(data.get("durations") or cleanup_data.get("durations") or [])
        lines = list(
            data.get("lines")
            or translation.get("lines")
            or tts.get("lines")
            or cleanup_data.get("lines")
            or []
        )
        transcript = await _load_cues(ctx, data)
        if not project_id or not source_video or not clean_cues:
            raise PipelineNodeError("RenderNode requires project_id, source_video and clean_cues")
        cues = _build_timeline_cues(
            transcript, lines or [Path(path).stem for path in clean_cues], durations
        )
        timing_mode = data.get("timing_mode", "Cue-Locked Audio Master Sync")
        timeline = TL.build_timeline(timing_mode, cues)
        outdir = edition_dir(project_id, target) / "exports" / data.get("export_version", "V1")
        outdir.mkdir(parents=True, exist_ok=True)
        silent = outdir / "video_silent.mp4"
        await ctx.update_progress(0.25, "rendering silent video")
        if data.get("dry_run"):
            silent.write_bytes(b"dry-run-video")
        else:
            concat = await asyncio.to_thread(
                EX.build_segments_concat,
                source_video,
                timeline,
                outdir / "work",
                data.get("render_workers", 3),
                data.get("fast_copy", False),
            )
            await asyncio.to_thread(EX.concat_video, concat, silent, True)
        timeline_path = outdir / "timeline.json"
        timeline_path.write_text(
            json.dumps(
                (
                    timeline.model_dump()
                    if hasattr(timeline, "model_dump")
                    else _timeline_to_dict(timeline)
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return NodeExecutionResult(
            node=self.name,
            data={
                "project_id": project_id,
                "target": target,
                "silent_video": str(silent),
                "timeline": _timeline_to_dict(timeline),
                "timeline_path": str(timeline_path),
                "outdir": str(outdir),
            },
        )


class ExportNode(PipelineNode):
    """Finalize MP4/export artifacts by wrapping existing exporter helpers."""

    def __init__(self, services=None) -> None:
        super().__init__("export", services)

    async def run(self, ctx: WorkflowContext) -> NodeExecutionResult:
        data = self.merged_inputs(ctx)
        deps = await self.all_dependency_results(ctx)
        render = _node_data(deps.get("render"))
        cleanup_data = _node_data(deps.get("audio_cleanup"))
        translation = _node_data(deps.get("translation")) or await _result_for(ctx, "translation")
        tts = _node_data(deps.get("tts")) or await _result_for(ctx, "tts")
        project_id = data.get("project_id") or render.get("project_id")
        target = str(
            data.get("target") or render.get("target") or cleanup_data.get("target") or "english"
        )
        outdir = Path(
            render.get("outdir")
            or edition_dir(str(project_id), target)
            / "exports"
            / str(data.get("export_version", "V1"))
        )
        outdir.mkdir(parents=True, exist_ok=True)
        silent = Path(str(data.get("silent_video") or render.get("silent_video") or ""))
        clean_cues = list(data.get("clean_cues") or cleanup_data.get("clean_cues") or [])
        lines = list(data.get("lines") or translation.get("lines") or tts.get("lines") or [])
        final = outdir / data.get("final_name", "final.mp4")
        audio_master = outdir / "audio_master.wav"
        await ctx.update_progress(0.2, "exporting")
        if data.get("dry_run"):
            audio_master.write_bytes(b"dry-run-audio")
            final.write_bytes(b"dry-run-final")
        else:
            await _write_audio_master(clean_cues, audio_master)
            await asyncio.to_thread(EX.mux_audio, silent, audio_master, final, None)
        script = outdir / "final_script.txt"
        script.write_text("\n".join(lines), encoding="utf-8")
        srt_path = None
        if data.get("captions") and render.get("timeline"):
            subs = SRT.retime_from_timeline(
                dict(enumerate(lines)), _timeline_from_dict(render["timeline"])
            )
            srt = outdir / "final.srt"
            SRT.write_srt(subs, str(srt))
            srt_path = str(srt)
        quality_path = outdir / "quality.json"
        report = {
            "project_id": project_id,
            "target": target,
            "cues": len(clean_cues),
            "final": str(final),
        }
        quality_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return NodeExecutionResult(
            node=self.name,
            data={
                "project_id": project_id,
                "target": target,
                "final": str(final),
                "audio_master": str(audio_master),
                "script": str(script),
                "srt": srt_path,
                "quality": str(quality_path),
            },
        )


# ---- helpers ----


async def _result_for(ctx: WorkflowContext, node_id: str) -> dict[str, Any]:
    run = await ctx.engine.require_run(ctx.run_id)
    state = run.node_states.get(node_id)
    if state is None or not isinstance(state.result, dict):
        return {}
    return _node_data(state.result)


def _require(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value in (None, ""):
        raise PipelineNodeError(f"required input missing: {key}")
    return value


def _node_data(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return result.get("data", result) if isinstance(result, dict) else {}


async def _load_cues(ctx: WorkflowContext, data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("transcript"):
        return list(data["transcript"])
    transcript_json = data.get("transcript_json")
    if not transcript_json:
        deps = await ctx.engine.require_run(ctx.run_id)
        for state in deps.node_states.values():
            if isinstance(state.result, dict):
                node_data = _node_data(state.result)
                transcript_json = transcript_json or node_data.get("transcript_json")
                if node_data.get("transcript"):
                    return list(node_data["transcript"])
    if transcript_json and Path(transcript_json).exists():
        return json.loads(Path(transcript_json).read_text(encoding="utf-8"))
    project_id = data.get("project_id")
    if project_id:
        path = project_dir(project_id) / "transcript" / "transcript.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return []


def _build_timeline_cues(
    transcript: list[dict[str, Any]], lines: list[str], durations: list[float]
) -> list[TL.Cue]:
    cues: list[TL.Cue] = []
    total = max(len(lines), len(durations), len(transcript))
    for i in range(total):
        seg = transcript[i] if i < len(transcript) else {"start": float(i), "end": float(i + 1)}
        text = lines[i] if i < len(lines) else str(seg.get("text", ""))
        cue = TL.Cue(
            idx=i,
            src_start=float(seg.get("start", i)),
            src_end=float(seg.get("end", i + 1)),
            text=text,
        )
        if i < len(durations):
            cue.audio_seconds = float(durations[i])
        cues.append(cue)
    return cues


def _timeline_to_dict(timeline: Any) -> dict[str, Any]:
    return {
        "total_seconds": getattr(timeline, "total_seconds", 0),
        "segments": [
            dict(segment.__dict__) if hasattr(segment, "__dict__") else dict(segment)
            for segment in getattr(timeline, "segments", [])
        ],
    }


def _timeline_from_dict(data: dict[str, Any]) -> Any:
    segments = []
    for raw in data.get("segments", []):
        segments.append(TL.Segment(**raw))
    return TL.Timeline(segments=segments, total_seconds=float(data.get("total_seconds", 0)))


async def _write_audio_master(clean_cues: list[str], audio_master: Path) -> None:
    def _write() -> None:
        import numpy as np
        import soundfile as sf

        with sf.SoundFile(
            str(audio_master), mode="w", samplerate=48000, channels=1, subtype="PCM_16"
        ) as handle:
            for cue in clean_cues:
                audio, _ = sf.read(cue)
                handle.write(np.asarray(audio, dtype="float32"))

    await asyncio.to_thread(_write)
