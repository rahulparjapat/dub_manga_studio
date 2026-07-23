from __future__ import annotations

from pathlib import Path
import hashlib
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ...ingest.upload import VIDEO_EXTS
from ..dependencies import get_state, get_storage
from ..schemas import OkResponse, UploadChunkResponse, UploadInitRequest

router = APIRouter(prefix="/uploads", tags=["uploads"])
META = "uploads:"


@router.post("/validate", response_model=OkResponse)
async def validate_upload(request: UploadInitRequest):
    suffix = Path(request.filename).suffix.lower()
    return OkResponse(data={"valid": suffix in VIDEO_EXTS, "supported_extensions": sorted(VIDEO_EXTS)})


@router.post("/init", response_model=dict, status_code=201)
async def init_upload(request: UploadInitRequest, state=Depends(get_state)):
    suffix = Path(request.filename).suffix.lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail="unsupported upload file type")
    upload_id = str(uuid4())
    root = state.upload_root / upload_id
    root.mkdir(parents=True, exist_ok=True)
    meta = {**request.model_dump(), "upload_id": upload_id, "received_bytes": 0, "complete": False, "sha256_actual": None}
    await state.storage.set_kv(META + upload_id, meta)
    return meta


@router.post("/{upload_id}/chunk", response_model=UploadChunkResponse)
async def upload_chunk(upload_id: str, chunk: UploadFile = File(...), state=Depends(get_state)):
    meta = await state.storage.get_kv(META + upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="upload not found")
    root = state.upload_root / upload_id
    root.mkdir(parents=True, exist_ok=True)
    chunk_path = root / f"chunk-{meta['received_bytes']:020d}.part"
    data = await chunk.read()
    chunk_path.write_bytes(data)
    meta["received_bytes"] = int(meta.get("received_bytes", 0)) + len(data)
    await state.storage.set_kv(META + upload_id, meta)
    return UploadChunkResponse(upload_id=upload_id, received_bytes=meta["received_bytes"], complete=False)


@router.post("/{upload_id}/complete", response_model=UploadChunkResponse)
async def complete_upload(upload_id: str, state=Depends(get_state)):
    meta = await state.storage.get_kv(META + upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="upload not found")
    root = state.upload_root / upload_id
    final = root / meta["filename"]
    digest = hashlib.sha256()
    with final.open("wb") as out:
        for part in sorted(root.glob("chunk-*.part")):
            data = part.read_bytes(); digest.update(data); out.write(data)
    actual = digest.hexdigest()
    if meta.get("sha256") and actual.lower() != str(meta["sha256"]).lower():
        raise HTTPException(status_code=400, detail="upload checksum mismatch")
    meta["sha256_actual"] = actual
    object_key = f"uploads/{upload_id}/{meta['filename']}"
    await state.storage.put_object(object_key, final.read_bytes(), content_type=meta.get("content_type"))
    meta.update({"complete": True, "object_key": object_key})
    await state.storage.set_kv(META + upload_id, meta)
    return UploadChunkResponse(upload_id=upload_id, received_bytes=meta["received_bytes"], complete=True, object_key=object_key)
