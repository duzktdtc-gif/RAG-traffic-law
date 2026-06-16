"""
vector_store.py
===============
Lưu trữ và tìm kiếm chunk embedding bằng ChromaDB (local persistent).

Hỗ trợ hai chế độ retrieval:
  1. Dense search   — cosine similarity trên embeddings
  2. Hybrid search  — kết hợp Dense + BM25 (Reciprocal Rank Fusion)

Schema ChromaDB collection:
    documents: chunk.text
    embeddings: np.ndarray (embed_dim,)
    metadatas: {
        doc_id, chunk_id, chunk_type,
        question, source_url,
        domain_tags (JSON string),
        has_refs, chunk_index, total_chunks
    }
    ids: chunk.chunk_id
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import pickle
import chromadb
from chromadb.config import Settings

from chunker import Chunk
from embedder import TrafficLawEmbedder


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

COLLECTION_NAME  = "traffic_law_vn"
DEFAULT_TOP_K    = 6
UPSERT_BATCH     = 512      # ChromaDB upsert batch size


# ─────────────────────────────────────────────
# Result schema
# ─────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id:   str
    doc_id:     str
    score:      float        # cosine similarity [0, 1]
    text:       str
    question:   str
    source_url: str
    chunk_type: str
    domain_tags: list[str]
    references:  list[dict]
    chunk_index: int
    total_chunks: int

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "doc_id":      self.doc_id,
            "score":       round(self.score, 4),
            "text":        self.text,
            "question":    self.question,
            "source_url":  self.source_url,
            "chunk_type":  self.chunk_type,
            "domain_tags": self.domain_tags,
            "references":  self.references,
            "chunk_index": self.chunk_index,
            "total_chunks":self.total_chunks,
        }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _meta_to_chroma(chunk: Chunk) -> dict:
    """Chuyển chunk.metadata thành dict chroma-compatible (no nested objects)."""
    m = chunk.metadata
    return {
        "doc_id":       chunk.doc_id,
        "chunk_type":   chunk.chunk_type,
        "question":     m.get("question", ""),
        "source_url":   m.get("source_url", ""),
        "domain_tags":  json.dumps(m.get("domain_tags", []), ensure_ascii=False),
        "has_refs":     int(m.get("has_refs", False)),
        "references":   json.dumps(m.get("references", []), ensure_ascii=False),
        "chunk_index":  m.get("chunk_index", 0),
        "total_chunks": m.get("total_chunks", 1),
        "token_count":  chunk.token_count,
    }


def _chroma_result_to_search(
    chunk_id: str,
    doc: str,
    meta: dict,
    distance: float,
) -> SearchResult:
    """Chuyển kết quả ChromaDB → SearchResult."""
    # ChromaDB dùng cosine distance [0,2], convert về similarity [0,1]
    score = 1.0 - distance / 2.0
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=meta.get("doc_id", ""),
        score=score,
        text=doc,
        question=meta.get("question", ""),
        source_url=meta.get("source_url", ""),
        chunk_type=meta.get("chunk_type", ""),
        domain_tags=json.loads(meta.get("domain_tags", "[]")),
        references=json.loads(meta.get("references", "[]")),
        chunk_index=meta.get("chunk_index", 0),
        total_chunks=meta.get("total_chunks", 1),
    )


def _reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    bm25_results:  list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) kết hợp dense + BM25 ranking.
    score_rrf = Σ 1/(k + rank_i)
    """
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(dense_results, start=1):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1 / (k + rank)
        result_map[r.chunk_id] = r

    for rank, r in enumerate(bm25_results, start=1):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1 / (k + rank)
        if r.chunk_id not in result_map:
            result_map[r.chunk_id] = r

    # Gán RRF score vào kết quả
    fused = []
    for cid, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        r = result_map[cid]
        fused.append(SearchResult(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            score=score,
            text=r.text,
            question=r.question,
            source_url=r.source_url,
            chunk_type=r.chunk_type,
            domain_tags=r.domain_tags,
            references=r.references,
            chunk_index=r.chunk_index,
            total_chunks=r.total_chunks,
        ))
    return fused


# ─────────────────────────────────────────────
# BM25 helper (in-memory, built on demand)
# ─────────────────────────────────────────────

class _BM25Index:
    """BM25 index nhẹ, thuần Python, không phụ thuộc thư viện ngoài."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus: list[str]         = []
        self.chunk_ids: list[str]      = []
        self._inverted: dict[str, list[int]] = {}
        self._dl: list[int]            = []
        self._avgdl: float             = 0.0
        self._tf: list[dict[str, int]] = []

    def _tokenize(self, text: str) -> list[str]:
        import re
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        # Bigrams để cải thiện recall tiếng Việt
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens + bigrams

    def build(self, texts: list[str], chunk_ids: list[str]):
        self.corpus    = texts
        self.chunk_ids = chunk_ids
        self._inverted = {}
        self._dl       = []

        for i, text in enumerate(texts):
            toks = self._tokenize(text)
            self._dl.append(len(toks))
            tf_doc: dict[str, int] = {}
            for tok in toks:
                tf_doc[tok] = tf_doc.get(tok, 0) + 1
            self._tf.append(tf_doc)

            for tok in tf_doc:
                self._inverted.setdefault(tok, []).append(i)

        self._avgdl = sum(self._dl) / len(self._dl) if self._dl else 1.0

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        import math
        N = len(self.corpus)
        if N == 0:
            return []

        scores: dict[int, float] = {}
        query_toks = self._tokenize(query)

        for tok in query_toks:
            postings = self._inverted.get(tok, [])
            df = len(postings)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for i in postings:
                dl  = self._dl[i]
                tf = self._tf[i].get(tok, 0)   # tra cứu TF đã pre-compute
                bm25 = idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
                scores[i] = scores.get(i, 0) + bm25

        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# ─────────────────────────────────────────────
# VectorStore class
# ─────────────────────────────────────────────

class TrafficLawVectorStore:
    """
    Vector store persistent dùng ChromaDB + BM25 cho hybrid search.

    Usage:
        store = TrafficLawVectorStore("./vectordb")
        store.build(chunks, embedder)          # build từ đầu
        results = store.search(query, embedder) # retrieval
    """

    def __init__(self, persist_dir: str = "./vectordb"):
        self.persist_dir = str(Path(persist_dir).resolve())
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._bm25: Optional[_BM25Index] = None

    # ── Init ──────────────────────────────────
    def _bm25_path(self) -> Path:
        return Path(self.persist_dir) / "bm25_index.pkl"

    def save_bm25(self, verbose=True):
        if self._bm25 is None:
            return
        with open(self._bm25_path(), "wb") as f:
            pickle.dump(self._bm25, f, protocol=5)

    def load_bm25_from_disk(self, verbose=True) -> bool:
        path = self._bm25_path()
        if not path.exists():
            return False
        with open(path, "rb") as f:
            self._bm25 = pickle.load(f)
        return True

    def _get_collection(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},   # cosine distance
            )
        return self._collection

    # ── Build ─────────────────────────────────

    def build(
        self,
        chunks: list[Chunk],
        embedder: TrafficLawEmbedder,
        clear_existing: bool = True,
        verbose: bool = True,
    ):
        """
        Embed toàn bộ chunks và upsert vào ChromaDB.
        Đồng thời build BM25 index in-memory.

        Args:
            chunks:         Danh sách Chunk từ chunker
            embedder:       TrafficLawEmbedder instance
            clear_existing: Xóa collection cũ trước khi build
            verbose:        In progress log
        """
        col = self._get_collection()

        if clear_existing and col.count() > 0:
            if verbose:
                print(f"  [VectorStore] Xóa {col.count():,} chunk cũ...")
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = None
            col = self._get_collection()

        if verbose:
            print(f"\n  [VectorStore] Build: {len(chunks):,} chunks → {self.persist_dir}")
            t_total = time.time()

        # ── Step 1: Embed ──────────────────────
        if embedder._use_tfidf and embedder._tfidf_vec is None:
            # Pre-fit TF-IDF trên full corpus text
            corpus_texts = [c.text for c in chunks]
            embedder.fit_tfidf(corpus_texts)

        _, vecs = embedder.embed_chunks(chunks, verbose=verbose)

        # ── Step 2: Upsert vào ChromaDB ────────
        if verbose:
            print(f"  [VectorStore] Upsert vào ChromaDB (batch={UPSERT_BATCH})...")

        t0 = time.time()
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch_chunks = chunks[i:i+UPSERT_BATCH]
            batch_vecs   = vecs[i:i+UPSERT_BATCH]

            col.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_vecs.tolist(),
                documents=[c.text for c in batch_chunks],
                metadatas=[_meta_to_chroma(c) for c in batch_chunks],
            )

            if verbose and (i // UPSERT_BATCH) % 5 == 0:
                print(f"    {i+len(batch_chunks):,}/{len(chunks):,} chunks upserted...")

        if verbose:
            print(f"  [VectorStore] Upsert done: {time.time()-t0:.1f}s")

        # ── Step 3: Build BM25 ─────────────────
        if verbose:
            print(f"  [VectorStore] Build BM25 index...")
        self._bm25 = _BM25Index()
        self._bm25.build(
            texts=[c.text for c in chunks],
            chunk_ids=[c.chunk_id for c in chunks],
        )

        if verbose:
            total_time = time.time() - t_total
            print(f"\n  [VectorStore] ✓ Build hoàn tất trong {total_time:.1f}s")
            print(f"  [VectorStore] Collection: {col.count():,} chunks")
            print(f"  [VectorStore] Persist dir: {self.persist_dir}")

    # ── Load existing ─────────────────────────

    def load_bm25_from_collection(self, verbose: bool = True):
        """
        Load lại BM25 index từ ChromaDB (khi restart, không cần rebuild).
        Tốn thời gian O(N) nhưng chỉ cần làm một lần.
        """
        if self.load_bm25_from_disk(verbose=verbose):
            return
        col = self._get_collection()
        n = col.count()
        if n == 0:
            if verbose:
                print("  [VectorStore] Collection rỗng, skip BM25 load.")
            return

        if verbose:
            print(f"  [VectorStore] Load BM25 từ {n:,} chunks trong ChromaDB...")

        result = col.get(include=["documents", "metadatas"])
        texts     = result["documents"]
        chunk_ids = result["ids"]

        self._bm25 = _BM25Index()
        self._bm25.build(texts, chunk_ids)
        if verbose:
            print(f"  [VectorStore] BM25 index loaded OK")

    # ── Search ────────────────────────────────

    def search_dense(
        self,
        query_vec: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        filter_domain: Optional[str] = None,
    ) -> list[SearchResult]:
        """Dense vector search với cosine similarity."""
        col = self._get_collection()

        where = None
        if filter_domain:
            # ChromaDB where clause (exact match trên string field)
            # domain_tags là JSON string, dùng $contains
            where = {"domain_tags": {"$contains": filter_domain}}

        result = col.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        results = []
        for i in range(len(result["ids"][0])):
            results.append(_chroma_result_to_search(
                chunk_id=result["ids"][0][i],
                doc=result["documents"][0][i],
                meta=result["metadatas"][0][i],
                distance=result["distances"][0][i],
            ))
        return results

    def search_bm25(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """BM25 keyword search."""
        if self._bm25 is None:
            self.load_bm25_from_collection(verbose=False)
        if self._bm25 is None:
            return []

        hits = self._bm25.search(query, top_k * 2)
        col  = self._get_collection()
        results = []
        max_bm25 = hits[0][1] if hits else 1.0

        for idx, bm25_score in hits[:top_k]:
            chunk_id = self._bm25.chunk_ids[idx]
            raw = col.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if not raw["ids"]:
                continue
            r = _chroma_result_to_search(
                chunk_id=chunk_id,
                doc=raw["documents"][0],
                meta=raw["metadatas"][0],
                distance=0.0,  # placeholder
            )
            r.score = bm25_score / max_bm25
            results.append(r)

        return results

    def search_hybrid(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        filter_domain: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: Dense + BM25 → Reciprocal Rank Fusion.

        Tốt nhất cho truy vấn pháp lý Việt Nam:
          - Dense giúp tìm ngữ nghĩa tương tự
          - BM25 giúp khớp từ khóa chính xác (điều luật, mức phạt)
        """
        dense_results = self.search_dense(query_vec, top_k * 2, filter_domain)
        bm25_results  = self.search_bm25(query, top_k * 2)
        fused         = _reciprocal_rank_fusion(dense_results, bm25_results)
        return fused[:top_k]

    # ── Stats ─────────────────────────────────

    def stats(self) -> dict:
        """Thống kê collection."""
        col = self._get_collection()
        count = col.count()
        return {
            "total_chunks": count,
            "persist_dir":  self.persist_dir,
            "collection":   COLLECTION_NAME,
            "bm25_ready":   self._bm25 is not None,
        }


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_documents
    from chunker import chunk_all_documents
    from embedder import TrafficLawEmbedder

    TEST_N = 100  # test nhanh với 100 docs

    print("=== VECTOR STORE TEST ===\n")

    # Load & chunk
    docs   = load_documents("/mnt/user-data/uploads/Dataset_6000.csv", verbose=False)
    chunks = chunk_all_documents(docs[:TEST_N], verbose=False)
    print(f"Chunks: {len(chunks)}")

    # Embed
    embedder = TrafficLawEmbedder()

    # Build store
    store = TrafficLawVectorStore("/home/claude/rag_traffic/vectordb_test")
    store.build(chunks, embedder, clear_existing=True, verbose=True)

    print(f"\nStats: {store.stats()}")

    # Test queries
    queries = [
        "vượt đèn đỏ bị phạt bao nhiêu?",
        "không có giấy đăng ký xe máy mức phạt",
        "độ tuổi được cấp giấy phép lái xe",
    ]

    for q in queries:
        qvec = embedder.embed_query(q)
        print(f"\n{'─'*55}")
        print(f"Query: {q}")
        print(f"{'─'*55}")

        # Dense
        print("\n[Dense Top-3]")
        for r in store.search_dense(qvec, top_k=3):
            print(f"  [{r.score:.4f}] {r.question[:60]}")

        # Hybrid
        print("\n[Hybrid Top-3]")
        for r in store.search_hybrid(q, qvec, top_k=3):
            print(f"  [{r.score:.4f}] {r.question[:60]}")
