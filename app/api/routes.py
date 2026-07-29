
from __future__ import annotations
import logging, os, shutil, tempfile, threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.cache.cache_service import CacheService
from src.connectors.file_connector import FileConnector
from src.embeddings.vector_store import QdrantVectorStore
from src.evaluation.metrics import MetricsTracker
from src.llms.models import ModelFactory
from src.parsers.pipeline import load_and_chunk
from src.rag_chain import RAGChain
from src.retrievers.pipeline import retrieve as pipeline_retrieve

logger = logging.getLogger(__name__)

_store = None
_feedback_records = []
_feedback_lock = threading.Lock()
_uploaded_sources = {}
_upload_lock = threading.Lock()

def _get_store():
    global _store
    if _store is None:
        _store = QdrantVectorStore()
    return _store

def _reset_store():
    global _store
    if _store is not None:
        try: _store.close()
        except Exception: pass
    _store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging; logging.getLogger(__name__).info("FastAPI starting")
    from src.utils.config import load_config; load_config()
    yield
    global _store
    if _store is not None:
        try: _store.close()
        except Exception: pass
    logging.getLogger(__name__).info("FastAPI shutting down")

app = FastAPI(title="Enterprise RAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    use_cache: bool = Field(True)

class AskResponse(BaseModel):
    question: str
    answer: str
    source_metadata: List[Dict[str, Any]]
    cached: bool
    model: Optional[str] = None
    latency_seconds: Optional[float] = None

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=100)

class RetrieveResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    count: int

class FeedbackRequest(BaseModel):
    query: str
    rating: str
    comment: Optional[str] = None
    answer: Optional[str] = None

class SwitchModelRequest(BaseModel):
    model_type: str
    provider: str
    model_name: Optional[str] = None

class DocumentInfo(BaseModel):
    source: str
    file_type: Optional[str] = None
    chunk_count: int = 0
    uploaded_at: Optional[str] = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    from src.utils.config import get_config
    cfg = get_config()
    suffix = Path(file.filename).suffix.lower().lstrip(".")
    allowed = getattr(cfg.document, "supported_formats", ["pdf", "docx", "txt", "md"])
    from src.parsers.smart_parser import ALLOWED_EXTENSIONS
    allowed = list(ALLOWED_EXTENSIONS)
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支持该文件类型 (.{suffix})，请选择规定类型的文件。允许: {', '.join(sorted(allowed))}"
        )
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        content = await file.read()
        with open(tmp_path, "wb") as fh:
            fh.write(content)
        connector = FileConnector(path=tmp_path)
        chunks = load_and_chunk(connector)
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted")
        store = _get_store()
        store.add_documents(chunks)
        source_name = Path(file.filename).name
        with _upload_lock:
            _uploaded_sources[source_name] = {"source": source_name, "file_type": suffix, "chunk_count": len(chunks), "uploaded_at": datetime.now(timezone.utc).isoformat()}
        return {"status": "ok", "filename": file.filename, "chunk_count": len(chunks), "message": "Indexed "+str(len(chunks))+" chunks"}
    except HTTPException:
        raise
    except Exception as exc:
        import logging; logging.getLogger(__name__).exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    tracker = MetricsTracker()
    tracker.start_request(req.question)
    try:
        chain = RAGChain(top_k=req.top_k, use_cache=req.use_cache)
        result = chain.answer(req.question)
    except Exception as exc:
        import logging; logging.getLogger(__name__).exception("RAG chain failed")
        raise HTTPException(status_code=500, detail="RAG processing failed: "+str(exc))
    num_retrieved = len(result.get("retrieved_docs", []))
    tracker.end_request(num_retrieved=num_retrieved)
    return AskResponse(
        question=result.get("question", req.question),
        answer=result.get("answer", ""),
        source_metadata=result.get("source_metadata", []),
        cached=result.get("cached", False),
        model=result.get("model"),
        latency_seconds=result.get("latency_seconds"),
    )

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_docs(req: RetrieveRequest):
    try:
        docs = pipeline_retrieve(query=req.query, top_k=req.top_k)
    except Exception as exc:
        import logging; logging.getLogger(__name__).exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed: "+str(exc))
    results = []
    for doc in docs:
        results.append({"content": doc.page_content, "source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", 1), "section_title": doc.metadata.get("section_title", ""), "score": doc.metadata.get("rerank_score", doc.metadata.get("qdrant_score", 0)), "file_type": doc.metadata.get("file_type", "")})
    return RetrieveResponse(query=req.query, results=results, count=len(results))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    docs = {}
    with _upload_lock:
        for src, info in _uploaded_sources.items():
            docs[src] = DocumentInfo(**info)
    try:
        from qdrant_client import QdrantClient
        from src.utils.config import get_config as _gc
        ccfg = _gc()
        client = QdrantClient(host=ccfg.qdrant.host, port=ccfg.qdrant.port)
        seen = set()
        offset = None
        while True:
            points, offset = client.scroll(collection_name=ccfg.qdrant.collection_name, limit=100, offset=offset, with_payload=["source", "file_type"], with_vectors=False)
            for pt in points:
                if src not in seen:
                    seen.add(src)
                    if src not in docs:
                        docs[src] = DocumentInfo(source=src, file_type=pt.payload.get("file_type") if pt.payload else None, chunk_count=1)
            if offset is None:
                break
    except Exception:
        pass
    return list(docs.values())

@app.delete("/documents/{source:path}")
async def delete_document(source: str):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from src.utils.config import get_config as _gc
        ccfg = _gc()
        client = QdrantClient(host=ccfg.qdrant.host, port=ccfg.qdrant.port)
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        client.delete(collection_name=ccfg.qdrant.collection_name, points_selector=flt)
        with _upload_lock:
            _uploaded_sources.pop(source, None)
        CacheService().clear()
        return {"status": "deleted", "source": source}
    except Exception as exc:
        import logging; logging.getLogger(__name__).exception("Delete failed")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    record = {"query": req.query, "rating": req.rating, "comment": req.comment, "answer": req.answer, "timestamp": datetime.now(timezone.utc).isoformat()}
    with _feedback_lock:
        _feedback_records.append(record)
        if len(_feedback_records) > 10000:
            _feedback_records[:] = _feedback_records[-10000:]
    return {"status": "ok", "recorded": True}

@app.get("/metrics")
async def get_metrics(limit: int = Query(default=20, ge=1, le=500)):
    tracker = MetricsTracker()
    summary = tracker.summary()
    history = tracker.get_history(limit=limit)
    with _feedback_lock:
        thumbs_up = sum(1 for r in _feedback_records if r.get("rating") == "thumbs_up")
        thumbs_down = sum(1 for r in _feedback_records if r.get("rating") == "thumbs_down")
        total_feedback = len(_feedback_records)
    return {"summary": summary, "feedback": {"total": total_feedback, "thumbs_up": thumbs_up, "thumbs_down": thumbs_down}, "history": history}

@app.post("/switch_model")
async def switch_model(req: SwitchModelRequest):
    valid = {"embedding", "llm", "reranker"}
    if req.model_type not in valid:
        raise HTTPException(status_code=400, detail="Invalid model_type")
    if req.model_type == "embedding":
        ModelFactory._embedding_cache.clear()
    elif req.model_type == "llm":
        ModelFactory._llm_cache.clear()
        try: CacheService().clear()
        except Exception: pass
    elif req.model_type == "reranker":
        ModelFactory._reranker_cache.clear()
    from src.utils.config import get_config as _gc
    cfg = _gc()
    if req.model_type == "embedding":
        object.__setattr__(cfg.models.embedding, "provider", req.provider)
        if req.model_name: object.__setattr__(cfg.models.embedding, "model_name", req.model_name)
        _reset_store()
    elif req.model_type == "llm":
        object.__setattr__(cfg.models.llm, "provider", req.provider)
        if req.model_name: object.__setattr__(cfg.models.llm, "model_name", req.model_name)
    elif req.model_type == "reranker":
        object.__setattr__(cfg.models.reranker, "provider", req.provider)
        if req.model_name: object.__setattr__(cfg.models.reranker, "model_name", req.model_name)
    return {"status": "ok", "model_type": req.model_type, "provider": req.provider}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
