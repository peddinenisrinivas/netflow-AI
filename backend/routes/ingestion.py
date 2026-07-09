"""
routes/ingestion.py — Section 1: PCAP Data Ingestion

POST /api/ingestion/upload            Upload .pcap/.pcapng file
GET  /api/ingestion/files             List uploaded files
GET  /api/ingestion/files/{id}/preview  Preview extracted features
DELETE /api/ingestion/files/{id}      Remove a file
"""

import os
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils.state import uploaded_files
from utils.pcap_parser import parse_pcap, FEATURE_COLUMNS

router = APIRouter()
UPLOAD_DIR = "uploads"
ALLOWED_EXT = {".pcap", ".pcapng", ".cap"}
MAX_SIZE_MB = 500


class FileInfo(BaseModel):
    id:            str
    filename:      str
    size_bytes:    int
    size_mb:       float
    uploaded_at:   str
    flow_count:    int
    feature_count: int


@router.post("/upload", response_model=FileInfo)
async def upload_pcap(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {ALLOWED_EXT}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    size_bytes = len(content)

    if size_bytes > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_SIZE_MB} MB.")

    with open(dest, "wb") as f:
        f.write(content)

    df = parse_pcap(dest)
    file_id = file.filename

    uploaded_files[file_id] = {
        "id":            file_id,
        "filename":      file.filename,
        "path":          dest,
        "size_bytes":    size_bytes,
        "size_mb":       round(size_bytes / (1024 * 1024), 2),
        "uploaded_at":   datetime.utcnow().isoformat(),
        "flow_count":    len(df),
        "feature_count": len(FEATURE_COLUMNS),
        "dataframe":     df,
    }

    return FileInfo(
        id=file_id,
        filename=file.filename,
        size_bytes=size_bytes,
        size_mb=round(size_bytes / (1024 * 1024), 2),
        uploaded_at=uploaded_files[file_id]["uploaded_at"],
        flow_count=len(df),
        feature_count=len(FEATURE_COLUMNS),
    )


@router.get("/files", response_model=List[FileInfo])
def list_files():
    return [
        FileInfo(
            id=v["id"], filename=v["filename"],
            size_bytes=v["size_bytes"], size_mb=v["size_mb"],
            uploaded_at=v["uploaded_at"], flow_count=v["flow_count"],
            feature_count=v["feature_count"],
        )
        for v in uploaded_files.values()
    ]


@router.delete("/files/{file_id}")
def delete_file(file_id: str):
    if file_id not in uploaded_files:
        raise HTTPException(404, "File not found")
    path = uploaded_files[file_id]["path"]
    if os.path.exists(path):
        os.remove(path)
    del uploaded_files[file_id]
    return {"detail": f"'{file_id}' deleted"}


@router.get("/files/{file_id}/preview")
def preview_features(file_id: str, rows: int = 10):
    if file_id not in uploaded_files:
        raise HTTPException(404, "File not found")
    df = uploaded_files[file_id]["dataframe"]
    return {"file_id": file_id, "rows": len(df.head(rows)),
            "data": df.head(rows).to_dict(orient="records")}
