"""
api.py
======
Bước 2: FastAPI server đóng gói toàn bộ RAG pipeline.

Endpoints:
  POST /ask              — hỏi đáp thông thường (JSON response)
  POST /ask/stream       — streaming SSE (token-by-token)
  GET  /health           — health check + system stats
  GET  /history/{sid}    — lấy lịch sử hội thoại theo session

Free APIs dùng:
  - Groq llama-3.3-70b-versatile (primary)
  - Groq llama-3.1-8b-instant    (fast/fallback)

Chạy:
  pip install fastapi uvicorn sse-starlette
  uvicorn src.api:app --reload --port 8000
"""

import os

# Load .env tự động
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    load_dotenv(_Path(__file__).parent.parent / ".env")
except ImportError:
    pass
import sys
import time
import uuid
import pickle
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Thêm src vào path ────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from embedder import TrafficLawEmbedder
from vector_store import TrafficLawVectorStore
from retriever import TrafficLawRetriever
from generator import TrafficLawGenerator

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CKPT_DIR   = ROOT.parent / "checkpoints"
DB_DIR     = ROOT.parent / "vectordb"
MAX_HISTORY_PER_SESSION = 20

# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────

app = FastAPI(
    title="RAG Luật Giao Thông Việt Nam",
    description="Hỏi đáp pháp luật giao thông đường bộ dựa trên 6,817 văn bản luật",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Production: đổi thành domain cụ thể
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global pipeline (lazy load khi request đầu tiên) ──
_pipeline: dict = {}
_session_history: dict = defaultdict(list)   # sid → list[{role, content}]


def get_pipeline():
    """Load và cache toàn bộ RAG pipeline."""
    if _pipeline:
        return _pipeline

    print("[API] Loading RAG pipeline...")
    t0 = time.time()

    # Embedder
    ckpt_path = CKPT_DIR / "chunks.pkl"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint không tồn tại: {ckpt_path}. Chạy indexing_pipeline.py trước.")

    with open(ckpt_path, "rb") as f:
        chunks = pickle.load(f)

    embedder = TrafficLawEmbedder()
    embedder.fit_tfidf([c.text for c in chunks])
    embedder._use_tfidf = True

    # Vector store
    store = TrafficLawVectorStore(str(DB_DIR))
    store.load_bm25_from_collection(verbose=False)

    _pipeline["retriever"]  = TrafficLawRetriever(store, embedder, top_k=6, rerank_top_k=4)
    _pipeline["generator"]  = TrafficLawGenerator(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        default_model="groq_strong",
    )
    _pipeline["n_chunks"]   = len(chunks)
    _pipeline["load_time"]  = time.time() - t0

    print(f"[API] Pipeline ready trong {_pipeline['load_time']:.1f}s")
    return _pipeline


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    query:       str             = Field(..., min_length=3, max_length=500,
                                        description="Câu hỏi về luật giao thông")
    session_id:  Optional[str]  = Field(None, description="ID phiên để giữ lịch sử")
    model:       Optional[str]  = Field("groq_strong",
                                        description="groq_strong | groq_fast | gemini")
    search_mode: Optional[str]  = Field("hybrid",
                                        description="hybrid | dense | bm25")
    top_k:       Optional[int]  = Field(4, ge=1, le=10,
                                        description="Số chunks tham khảo")


class SourceDoc(BaseModel):
    question:   str
    source_url: str
    score:      float
    chunk_type: str
    references: list[dict]


class AskResponse(BaseModel):
    answer:       str
    session_id:   str
    model_used:   str
    citations:    list[str]
    sources:      list[SourceDoc]
    retrieval_ms: float
    generate_ms:  float
    total_ms:     float
    input_tokens: int
    output_tokens: int


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pre-load pipeline khi server khởi động."""
    get_pipeline()


@app.get("/health")
async def health():
    """Health check + thống kê hệ thống."""
    p = get_pipeline()
    return {
        "status":    "ok",
        "n_chunks":  p["n_chunks"],
        "load_time": round(p["load_time"], 2),
        "models": {
            "primary":  "groq/llama-3.3-70b-versatile",
            "fast":     "groq/llama-3.1-8b-instant",
            "fallback": "gemini/gemini-2.0-flash",
        },
        "endpoints": ["/ask", "/ask/stream", "/history/{session_id}"],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Hỏi đáp thông thường — trả về JSON đầy đủ.

    Ví dụ:
        curl -X POST http://localhost:8000/ask \\
          -H "Content-Type: application/json" \\
          -d '{"query": "Vượt đèn đỏ bị phạt bao nhiêu?"}'
    """
    t_total = time.time()
    p       = get_pipeline()

    # Session ID
    sid = req.session_id or str(uuid.uuid4())

    # ── Lịch sử hội thoại (nếu có) ───────────
    # Inject 2 turn gần nhất vào query để LLM hiểu ngữ cảnh
    history = _session_history[sid][-4:]   # 2 turns (user + bot)
    history_ctx = ""
    if history:
        history_ctx = "\n\nLịch sử hội thoại gần nhất:\n"
        for h in history:
            role = "Người dùng" if h["role"] == "user" else "Trợ lý"
            history_ctx += f"{role}: {h['content'][:200]}\n"

    # ── Retrieve ──────────────────────────────
    ret = p["retriever"].retrieve(
        req.query,
        search_mode=req.search_mode,
        verbose=False,
    )

    context = ret.context_text
    if history_ctx:
        context = history_ctx + "\n---\n\n" + context

    # ── Generate ──────────────────────────────
    gen = p["generator"].generate(
        query=req.query,
        context=context,
        model=req.model,
    )

    # ── Lưu lịch sử ──────────────────────────
    _session_history[sid].append({"role": "user",      "content": req.query})
    _session_history[sid].append({"role": "assistant",  "content": gen.answer})
    if len(_session_history[sid]) > MAX_HISTORY_PER_SESSION * 2:
        _session_history[sid] = _session_history[sid][-MAX_HISTORY_PER_SESSION * 2:]

    total_ms = (time.time() - t_total) * 1000

    return AskResponse(
        answer=gen.answer,
        session_id=sid,
        model_used=gen.model_used,
        citations=gen.citations,
        sources=[
            SourceDoc(
                question=c.question,
                source_url=c.source_url,
                score=round(c.score, 4),
                chunk_type=c.chunk_type,
                references=c.references,
            )
            for c in ret.chunks
        ],
        retrieval_ms=round(ret.retrieval_ms, 1),
        generate_ms=round(gen.latency_ms, 1),
        total_ms=round(total_ms, 1),
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming endpoint — trả về Server-Sent Events (SSE).
    Mỗi event là một token, client nhận và hiển thị real-time.

    Format SSE:
        data: <token>\\n\\n
        data: [DONE]\\n\\n

    Dùng với JavaScript:
        const es = new EventSource('/ask/stream?...');
        es.onmessage = (e) => {
            if (e.data === '[DONE]') es.close();
            else output += e.data;
        };
    """
    p   = get_pipeline()
    sid = req.session_id or str(uuid.uuid4())

    # Retrieve trước
    ret     = p["retriever"].retrieve(req.query, search_mode=req.search_mode, verbose=False)
    context = ret.context_text

    def token_stream():
        full_answer = []
        try:
            for token in p["generator"].generate_stream(
                query=req.query,
                context=context,
                model=req.model,
            ):
                full_answer.append(token)
                # SSE format
                yield f"data: {token}\n\n"

            # Gửi metadata khi xong
            import json
            from generator import extract_citations
            answer_text = "".join(full_answer)
            citations   = extract_citations(answer_text)
            meta = json.dumps({
                "session_id":   sid,
                "citations":    citations,
                "retrieval_ms": round(ret.retrieval_ms, 1),
                "sources": [
                    {"question": c.question, "url": c.source_url, "score": round(c.score, 4)}
                    for c in ret.chunks
                ],
            }, ensure_ascii=False)
            yield f"data: [META]{meta}\n\n"
            yield "data: [DONE]\n\n"

            # Lưu history
            _session_history[sid].append({"role": "user",      "content": req.query})
            _session_history[sid].append({"role": "assistant",  "content": answer_text})

        except Exception as e:
            yield f"data: [ERROR]{str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Lấy lịch sử hội thoại của một session."""
    history = _session_history.get(session_id, [])
    return {
        "session_id": session_id,
        "turns":      len(history) // 2,
        "history":    history,
    }


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Xóa lịch sử một session."""
    _session_history.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
