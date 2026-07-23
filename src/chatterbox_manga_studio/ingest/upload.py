"""Ingest: chunked upload handling, input-folder, Drive (gdown), fallback upload.

Gradio handles the actual browser chunking + %/cancel in the UI layer; this module
does the server-side receive, partial cleanup, listing, and Drive download.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import time
from pathlib import Path
from ..common.paths import INPUT, UPLOADS, project_dir
from ..common.logging_util import get_logger

log = get_logger("ingest")

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v"}


def list_input_videos() -> list[str]:
    INPUT.mkdir(parents=True, exist_ok=True)
    return sorted(str(p) for p in INPUT.iterdir()
                  if p.suffix.lower() in VIDEO_EXTS)


def check_input_video_ready(path: str | Path, min_stable_seconds: float = 6.0) -> dict:
    """Verify that an input video is fully copied and transcribable.

    Extension and file existence are not enough for multi-GB uploads. We require
    a stable modification time, a successful ffprobe parse, positive duration, and
    both video and audio streams before it may be auto-ingested.
    """
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
        return {"ok": False, "message": "Not a supported video file."}
    stat = p.stat()
    age = time.time() - stat.st_mtime
    if stat.st_size < 1024 or age < min_stable_seconds:
        return {"ok": False, "waiting": True,
                "message": f"Still uploading/copying ({max(0, min_stable_seconds - age):.0f}s stability wait)."}
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-show_streams", "-of", "json", str(p)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
    except FileNotFoundError:
        return {"ok": False, "message": "ffprobe is required to verify uploaded video; install ffmpeg."}
    except subprocess.TimeoutExpired:
        return {"ok": False, "waiting": True, "message": "Video probe timed out; file may still be copying."}
    if proc.returncode != 0:
        return {"ok": False, "waiting": True,
                "message": "Video is not readable yet; waiting for upload/copy to finish."}
    try:
        data = json.loads(proc.stdout or "{}")
        duration = float((data.get("format") or {}).get("duration") or 0)
        kinds = {s.get("codec_type") for s in data.get("streams", [])}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "waiting": True, "message": "Video metadata is incomplete; waiting."}
    if duration <= 0 or "video" not in kinds or "audio" not in kinds:
        return {"ok": False, "message": "Video must contain readable video and audio streams."}
    return {"ok": True, "path": str(p), "duration_s": duration,
            "message": f"Ready: {p.name} ({duration / 60:.1f} min, {stat.st_size / 2**30:.2f} GB)."}


def store_uploaded(tmp_path: str, project_id: str, filename: str) -> str:
    """Move a completed (Gradio) upload into the project's source folder."""
    dst_dir = project_dir(project_id) / "source"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename
    shutil.move(tmp_path, dst)
    return str(dst)


def auto_ingest_stable_input(project_id: str, min_stable_seconds: float = 6.0) -> dict:
    """Ingest the newest completed video from ``data/input``.

    A file is eligible only after it has not been modified for ``min_stable_seconds``.
    Copying uses a temporary name then an atomic replace, so transcription/export can
    never observe a half-copied source. Existing identical project sources are left
    untouched, making the UI timer idempotent.
    """
    INPUT.mkdir(parents=True, exist_ok=True)
    candidates = sorted((p for p in INPUT.iterdir()
                         if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
                        key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not candidates:
        return {"ok": False, "waiting": True,
                "message": "Waiting for a completed video in data/input/."}
    ready = None
    last_note = "Waiting for a completed video in data/input/."
    for candidate in candidates:
        check = check_input_video_ready(candidate, min_stable_seconds=min_stable_seconds)
        if check.get("ok"):
            ready = candidate
            break
        last_note = f"{candidate.name}: {check.get('message', last_note)}"
    if ready is None:
        return {"ok": False, "waiting": True, "message": last_note}
    src = ready
    dst_dir = project_dir(project_id) / "source"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return {"ok": True, "already": True, "path": str(dst),
                "message": f"Already ingested: {src.name}"}
    tmp = dst.with_name(f".{dst.name}.copying")
    try:
        shutil.copy2(src, tmp)
        if tmp.stat().st_size != src.stat().st_size:
            raise OSError("copied size did not match source")
        tmp.replace(dst)
        log.info("auto-ingested stable input %s -> %s", src, dst)
        return {"ok": True, "path": str(dst),
                "message": f"Auto-ingested completed input: {src.name}"}
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"ok": False, "message": f"Auto-ingest failed: {e}"}


def cleanup_partial(project_id: str) -> int:
    """Remove partial upload fragments for a project."""
    n = 0
    for p in UPLOADS.glob(f"{project_id}.*"):
        try:
            p.unlink(); n += 1
        except Exception:
            pass
    return n


def download_drive(url: str, project_id: str) -> dict:
    """Download a Google Drive video via gdown into project source."""
    try:
        import gdown
    except Exception:
        return {"ok": False, "error": "gdown not installed in app env."}
    dst_dir = project_dir(project_id) / "source"
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        out = gdown.download(url=url, output=str(dst_dir) + "/", fuzzy=True, quiet=False)
        if not out:
            return {"ok": False, "error": "gdown returned no file (check share link)."}
        return {"ok": True, "path": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}
