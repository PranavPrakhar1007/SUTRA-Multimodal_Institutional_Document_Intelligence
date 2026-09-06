"""FastAPI application for the NIT Jamshedpur Document RAG prototype."""

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import backend.config as cfg
from backend.config import (
    EMBEDDINGS_DIR,
    HYBRID_CANDIDATE_UNION_SIZE,
    HYBRID_RRF_TEXT_WEIGHT,
    HYBRID_RRF_VISUAL_WEIGHT,
    METADATA_DIR,
    PAGES_DIR,
    PROCESSED_DIR,
    VALID_MODES,
)
from backend.generation import generate_answer
from backend.hybrid_reranker import HybridRerankedRetriever
from backend.hybrid_retrieval import HybridRetriever
from backend.path_utils import resolve_page_image
from backend.text_retrieval import TextIndex
from backend.visual_reranker import VisualReranker
from backend.visual_retrieval import VisualIndex

app = FastAPI(
    title="NIT Jamshedpur Document Intelligence",
    description="Text, Visual and Hybrid RAG proof-of-concept over real NIT Jamshedpur documents",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_index: Optional[TextIndex] = None
visual_index: Optional[VisualIndex] = None
hybrid_retriever: Optional[HybridRetriever] = None
visual_reranker: Optional[VisualReranker] = None
hybrid_reranked_retriever: Optional[HybridRerankedRetriever] = None


@app.on_event("startup")
def startup():
    global text_index, visual_index, hybrid_retriever, visual_reranker, hybrid_reranked_retriever

    text_index = TextIndex()
    text_path = EMBEDDINGS_DIR / "text"
    if (text_path / "dense_embeddings.npy").exists():
        text_index.load(text_path)
    else:
        text_index.build_from_processed(PROCESSED_DIR)
        text_index.save(text_path)

    visual_index = VisualIndex()
    visual_path = EMBEDDINGS_DIR / "visual"
    if (visual_path / "visual_embeddings.npy").exists():
        visual_index.load(visual_path)
    else:
        visual_index.build_from_processed(PROCESSED_DIR)
        visual_index.save(visual_path)

    hybrid_retriever = HybridRetriever(text_index, visual_index)
    visual_reranker = VisualReranker()
    hybrid_reranked_retriever = HybridRerankedRetriever(text_index, visual_index)
    print(f"Ready: {len(text_index.entries)} text pages / {len(visual_index.entries)} visual pages")


ModeLiteral = Literal[
    "text", "text_baseline", "visual", "vision", "visual_colqwen2", "visual_clip_baseline", "visual_reranked",
    "hybrid", "hybrid_rrf_baseline", "hybrid_reranked"
]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: ModeLiteral = "hybrid_reranked"
    top_k: int = Field(default=5, ge=1, le=25)


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class SetKeyRequest(BaseModel):
    provider: Optional[str] = "groq"
    api_key: str = Field(min_length=10, max_length=500)


def _require_indexes():
    if text_index is None or visual_index is None or hybrid_retriever is None or visual_reranker is None or hybrid_reranked_retriever is None:
        raise HTTPException(status_code=503, detail="Retrieval indices are not loaded yet")


def _retrieve(question: str, mode: str, top_k: int):
    _require_indexes()
    if mode in ("text", "text_baseline"):
        return text_index.search(question, top_k=top_k)
    if mode in ("visual_reranked",):
        return visual_reranker.score_candidates(question, candidates=visual_index.search(question, top_k=top_k * 2), top_k=top_k)
    if mode in ("visual", "vision", "visual_colqwen2", "visual_clip_baseline"):
        return visual_index.search(question, top_k=top_k)
    if mode in ("hybrid_reranked",):
        return hybrid_reranked_retriever.search(question, top_k=top_k)
    return hybrid_retriever.search(
        question,
        top_k=top_k,
        text_weight=HYBRID_RRF_TEXT_WEIGHT,
        visual_weight=HYBRID_RRF_VISUAL_WEIGHT,
    )


def _page_response(r: dict) -> dict:
    nid = r.get("notice_id", r.get("doc_id", ""))
    pnum = int(r.get("page_number", r.get("page", 1)))
    page = {
        "notice_id": nid,
        "doc_id": r.get("doc_id", nid),
        "filename": r.get("filename", f"{nid}.pdf"),
        "file_name": r.get("file_name", f"{nid}.pdf"),
        "page_number": pnum,
        "page": pnum,
        "score": round(float(r.get("score", 0.0)), 6),
        "retrieval_method": r.get("retrieval_method", ""),
        "has_text": r.get("has_text", False),
        "image_url": f"/api/pages/{nid}/{pnum}",
    }
    for key in ("best_tile_box", "best_tile_label", "visual_evidence_type", "full_page_sim", "max_tile_sim", "fusion_details"):
        if key in r:
            page[key] = r[key]
    return page


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "text_index_loaded": text_index is not None and text_index.dense_embeddings is not None,
        "visual_index_loaded": visual_index is not None and visual_index.image_embeddings is not None,
        "total_pages": len(text_index.entries) if text_index else 0,
        "llm_provider": "groq",
        "configured_model": cfg.GROQ_MODEL,
    }


@app.post("/api/set-key")
def set_api_key(request: SetKeyRequest):
    import json
    import os

    cfg.GROQ_API_KEY = request.api_key
    os.environ["GROQ_API_KEY"] = request.api_key
    cfg.LLM_PROVIDER = "groq"

    try:
        keys_file = cfg.DATA_DIR / "keys.json"
        keys_data = {}
        if keys_file.exists():
            try:
                keys_data = json.loads(keys_file.read_text(encoding="utf-8"))
            except Exception:
                keys_data = {}
        keys_data["groq"] = request.api_key
        keys_data["api_key"] = request.api_key
        keys_data["provider"] = "groq"
        keys_file.write_text(json.dumps(keys_data, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"status": "ok", "provider": "groq", "message": "Groq API key set for this server session"}


@app.get("/api/models")
def list_models():
    """Return the configured Groq model."""
    return {"provider": "groq", "models": [cfg.GROQ_MODEL]}



@app.get("/api/documents")
def list_documents():
    manifest_path = project_root / "CORPUS_MANIFEST.csv"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = {row["notice_id"]: row for row in csv.DictReader(f)}

    documents = []
    for meta_file in sorted(METADATA_DIR.glob("notice_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        nid = meta["notice_id"]
        documents.append({
            "notice_id": nid,
            "canonical_filename": meta.get("canonical_filename", ""),
            "original_filename": manifest.get(nid, {}).get("original_filename", ""),
            "page_count": meta.get("page_count", len(meta.get("pages", []))),
            "text_availability": meta.get("text_availability", "unknown"),
        })
    return {"documents": documents, "total": len(documents)}


@app.get("/api/pages/{notice_id}/{page_number}")
def get_page_image(notice_id: str, page_number: int):
    if not re.fullmatch(r"notice_\d+", notice_id) or page_number < 1:
        raise HTTPException(status_code=400, detail="Invalid document/page identifier")
    image_path = resolve_page_image(notice_id, page_number)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(str(image_path), media_type="image/png")


@app.get("/api/corpus-profile")
def get_corpus_profile():
    path = project_root / "data" / "corpus_profile.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Corpus profile not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/query")
def query(request: QueryRequest):
    question = request.question.strip()
    mode = request.mode.lower()

    t_retrieval = time.perf_counter()
    results = _retrieve(question, mode, request.top_k)
    retrieval_time_ms = round((time.perf_counter() - t_retrieval) * 1000, 2)

    if not results:
        return {
            "question": question,
            "mode": mode,
            "answer": "No relevant document candidates were retrieved.",
            "source": None,
            "retrieved_pages": [],
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": 0,
            "total_time_ms": retrieval_time_ms,
            "status": "no_results",
        }

    top_result = results[0]
    t_generation = time.perf_counter()
    gen_result = generate_answer(question, top_result, mode=mode)
    generation_time_ms = round((time.perf_counter() - t_generation) * 1000, 2)

    retrieved_pages = [_page_response(r) for r in results]
    return {
        "question": question,
        "mode": mode,
        "answer": gen_result.get("answer", ""),
        "source": gen_result.get("source", {}),
        "generation_method": gen_result.get("generation_method", ""),
        "retrieved_pages": retrieved_pages,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_time_ms": round(retrieval_time_ms + generation_time_ms, 2),
        "status": gen_result.get("status", "unknown"),
    }


@app.post("/api/compare")
def compare(request: CompareRequest):
    question = request.question.strip()
    modes = [
        "text_baseline",
        "visual",
        "hybrid_rrf_baseline",
    ]
    comparison = {"question": question, "modes": {}}

    t_total = time.perf_counter()

    for mode in modes:
        t_retrieval = time.perf_counter()
        results = _retrieve(question, mode, request.top_k)
        retrieval_time_ms = round((time.perf_counter() - t_retrieval) * 1000, 2)

        if results:
            t_generation = time.perf_counter()
            gen = generate_answer(question, results[0], mode=mode)
            generation_time_ms = round((time.perf_counter() - t_generation) * 1000, 2)
            payload = {
                "answer": gen.get("answer", ""),
                "source": gen.get("source", {}),
                "generation_method": gen.get("generation_method", ""),
                "status": gen.get("status", "unknown"),
            }
        else:
            generation_time_ms = 0
            payload = {"answer": "No relevant document candidates were retrieved.", "source": None, "generation_method": "none", "status": "no_results"}

        payload["retrieved_pages"] = [_page_response(r) for r in results]
        payload["retrieval_time_ms"] = retrieval_time_ms
        payload["generation_time_ms"] = generation_time_ms
        comparison["modes"][mode] = payload

    comparison["total_time_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
    return comparison


frontend_dist = project_root / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
