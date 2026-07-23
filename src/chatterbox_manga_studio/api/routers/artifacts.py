from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ...common.paths import PROJECT_ROOT
from ..dependencies import get_storage

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("/download")
async def download_artifact(
    object_key: str | None = Query(None),
    path: str | None = Query(None),
    storage=Depends(get_storage),
):
    """Download an exported artifact by StorageManager object key or safe data path.

    Filesystem paths are restricted to the repository data directory to prevent
    traversal or arbitrary host-file exposure.
    """

    if object_key:
        try:
            data, meta = await storage.get_object(object_key)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail="artifact object not found") from exc
        return Response(
            content=data,
            media_type=meta.content_type or "application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{Path(object_key).name}"'},
        )
    if path:
        candidate = Path(path).expanduser().resolve()
        data_root = (PROJECT_ROOT / "data").resolve()
        try:
            candidate.relative_to(data_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="artifact path is outside data directory") from exc
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="artifact file not found")
        return FileResponse(candidate, filename=candidate.name)
    raise HTTPException(status_code=400, detail="object_key or path is required")
