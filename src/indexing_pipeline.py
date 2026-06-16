"""
indexing_pipeline.py
====================
Orchestrate toàn bộ quá trình:
    CSV → DataLoader → Chunker → Embedder → VectorStore

Hỗ trợ:
  - Full build từ đầu
  - Incremental update (thêm doc mới)
  - Checkpoint: lưu chunks và embeddings ra .npz để không cần re-embed
  - Báo cáo chi tiết từng bước

Dùng:
    # Full build
    python indexing_pipeline.py --csv Dataset_6000.csv

    # Dùng checkpoint (nếu đã build trước)
    python indexing_pipeline.py --csv Dataset_6000.csv --from-checkpoint
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_documents, Document
from chunker import chunk_all_documents, Chunk
from embedder import TrafficLawEmbedder, DEFAULT_MODEL
from vector_store import TrafficLawVectorStore


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_CSV      = "/mnt/user-data/uploads/Dataset_6000.csv"
DEFAULT_DB_DIR   = "./vectordb"
DEFAULT_CKPT_DIR = "./checkpoints"


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def save_checkpoint(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    ckpt_dir: str,
    verbose: bool = True,
):
    """
    Lưu chunks + embeddings ra disk để tái sử dụng.
    Format:
        chunks.pkl      — list[Chunk] (pickle)
        embeddings.npz  — np.ndarray float32
        checkpoint.json — metadata
    """
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)

    # Chunks
    with open(p / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f, protocol=5)

    # Embeddings
    np.savez_compressed(p / "embeddings.npz", embeddings=embeddings)

    # Metadata
    meta = {
        "n_chunks":   len(chunks),
        "embed_dim":  embeddings.shape[1],
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_types": {
            ct: sum(1 for c in chunks if c.chunk_type == ct)
            for ct in set(c.chunk_type for c in chunks)
        },
    }
    with open(p / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if verbose:
        size_chunks = (p / "chunks.pkl").stat().st_size / 1024
        size_embs   = (p / "embeddings.npz").stat().st_size / 1024
        print(f"  [Checkpoint] Saved → {ckpt_dir}")
        print(f"    chunks.pkl:      {size_chunks:.0f} KB")
        print(f"    embeddings.npz:  {size_embs:.0f} KB")


def load_checkpoint(
    ckpt_dir: str,
    verbose: bool = True,
) -> tuple[list[Chunk], np.ndarray] | None:
    """
    Load chunks + embeddings từ checkpoint.
    Trả về None nếu không tìm thấy.
    """
    p = Path(ckpt_dir)
    if not (p / "chunks.pkl").exists() or not (p / "embeddings.npz").exists():
        return None

    if verbose:
        print(f"  [Checkpoint] Loading từ {ckpt_dir}...")

    with open(p / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    data = np.load(p / "embeddings.npz")
    embeddings = data["embeddings"]

    if verbose:
        meta_path = p / "checkpoint.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  [Checkpoint] {meta['n_chunks']:,} chunks, dim={meta['embed_dim']}")
            print(f"  [Checkpoint] Tạo lúc: {meta['timestamp']}")

    return chunks, embeddings


# ─────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────

def step_load(csv_path: str, verbose: bool = True) -> list[Document]:
    print("\n" + "═"*60)
    print("  BƯỚC 1: LOAD & VALIDATE DATA")
    print("═"*60)
    t0 = time.time()
    docs = load_documents(csv_path, drop_duplicates=True, verbose=verbose)
    print(f"\n  ⏱ Hoàn thành trong {time.time()-t0:.2f}s")
    return docs


def step_chunk(docs: list[Document], verbose: bool = True) -> list[Chunk]:
    print("\n" + "═"*60)
    print("  BƯỚC 2: CHUNKING")
    print("═"*60)
    t0 = time.time()
    chunks = chunk_all_documents(docs, verbose=verbose)
    print(f"\n  ⏱ Hoàn thành trong {time.time()-t0:.2f}s")
    return chunks


def step_embed(
    chunks: list[Chunk],
    model_name: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> tuple[TrafficLawEmbedder, np.ndarray]:
    print("\n" + "═"*60)
    print("  BƯỚC 3: EMBEDDING")
    print("═"*60)
    print(f"  Model target: {model_name}")

    embedder = TrafficLawEmbedder(model_name=model_name)

    # Pre-fit TF-IDF trên full corpus nếu dùng fallback
    # (cần biết trước khi embed để có vocab đầy đủ)
    #corpus_texts = [c.text for c in chunks]

    t0 = time.time()
    _, vecs = embedder.embed_chunks(chunks, verbose=verbose)
    print(f"\n  ⏱ Hoàn thành trong {time.time()-t0:.2f}s")
    return embedder, vecs


def step_index(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    embedder: TrafficLawEmbedder,
    db_dir: str,
    verbose: bool = True,
) -> TrafficLawVectorStore:
    print("\n" + "═"*60)
    print("  BƯỚC 4: INDEX VÀO VECTOR DATABASE")
    print("═"*60)
    print(f"  Persist dir: {db_dir}")

    store = TrafficLawVectorStore(db_dir)

    # Upsert pre-computed embeddings vào ChromaDB
    # (không cần embed lại)
    t0 = time.time()

    col = store._get_collection()
    col_count = col.count()
    if col_count > 0:
        print(f"  Xóa {col_count:,} chunk cũ...")
        store._client.delete_collection("traffic_law_vn")
        store._collection = None
        col = store._get_collection()

    from vector_store import UPSERT_BATCH, _meta_to_chroma
    print(f"  Upsert {len(chunks):,} chunks (batch={UPSERT_BATCH})...")
    for i in range(0, len(chunks), UPSERT_BATCH):
        b_chunks = chunks[i:i+UPSERT_BATCH]
        b_vecs   = embeddings[i:i+UPSERT_BATCH]
        col.upsert(
            ids=[c.chunk_id for c in b_chunks],
            embeddings=b_vecs.tolist(),
            documents=[c.text for c in b_chunks],
            metadatas=[_meta_to_chroma(c) for c in b_chunks],
        )
        if verbose:
            pct = (i + len(b_chunks)) * 100 // len(chunks)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r    [{bar}] {i+len(b_chunks):,}/{len(chunks):,}", end="", flush=True)

    print(f"\n  Upsert done: {time.time()-t0:.1f}s")

    # Build BM25
    print("  Build BM25 index...")
    from vector_store import _BM25Index
    store._bm25 = _BM25Index()
    store._bm25.build(
        texts=[c.text for c in chunks],
        chunk_ids=[c.chunk_id for c in chunks],
    )

    print(f"\n  ⏱ Hoàn thành trong {time.time()-t0:.2f}s")
    return store


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    csv_path:      str  = DEFAULT_CSV,
    db_dir:        str  = DEFAULT_DB_DIR,
    ckpt_dir:      str  = DEFAULT_CKPT_DIR,
    model_name:    str  = DEFAULT_MODEL,
    from_checkpoint: bool = False,
    save_ckpt:     bool = True,
    verbose:       bool = True,
) -> TrafficLawVectorStore:
    """
    Chạy toàn bộ indexing pipeline.

    Args:
        csv_path:        Đường dẫn Dataset CSV
        db_dir:          Thư mục lưu ChromaDB
        ckpt_dir:        Thư mục checkpoint (chunks + embeddings)
        model_name:      HuggingFace model ID cho embedder
        from_checkpoint: Nếu True, load chunks+embs từ checkpoint (bỏ qua re-embed)
        save_ckpt:       Lưu checkpoint sau khi embed
        verbose:         In log chi tiết

    Returns:
        TrafficLawVectorStore đã sẵn sàng query
    """
    t_pipeline = time.time()

    print("\n" + "█"*60)
    print("  RAG LUẬT GIAO THÔNG VN — INDEXING PIPELINE")
    print("█"*60)
    print(f"  CSV     : {csv_path}")
    print(f"  DB      : {db_dir}")
    print(f"  Model   : {model_name}")

    embedder = TrafficLawEmbedder(model_name=model_name)

    if from_checkpoint:
        # ── Load từ checkpoint ─────────────────
        ckpt = load_checkpoint(ckpt_dir, verbose=verbose)
        if ckpt:
            chunks, embeddings = ckpt
            print(f"  Checkpoint loaded: {len(chunks):,} chunks")
        else:
            print("  ⚠ Không tìm thấy checkpoint, chạy full pipeline...")
            from_checkpoint = False

    if not from_checkpoint:
        # ── Full pipeline ──────────────────────
        docs       = step_load(csv_path, verbose=verbose)
        chunks     = step_chunk(docs, verbose=verbose)
        embedder, embeddings = step_embed(chunks, model_name=model_name, verbose=verbose)

        if save_ckpt:
            print(f"\n  Lưu checkpoint → {ckpt_dir}")
            save_checkpoint(chunks, embeddings, ckpt_dir, verbose=verbose)

    store = step_index(chunks, embeddings, embedder, db_dir, verbose=verbose)

    # ── Final summary ──────────────────────────
    total_time = time.time() - t_pipeline
    stats = store.stats()

    print("\n" + "█"*60)
    print("  PIPELINE HOÀN TẤT")
    print("█"*60)
    print(f"  Tổng thời gian : {total_time:.1f}s")
    print(f"  Chunks indexed : {stats['total_chunks']:,}")
    print(f"  Vector DB      : {stats['persist_dir']}")
    print(f"  BM25 ready     : {stats['bm25_ready']}")
    print("█"*60 + "\n")

    return store


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Traffic Law VN — Indexing Pipeline"
    )
    parser.add_argument("--csv",  default=DEFAULT_CSV,    help="Đường dẫn dataset CSV")
    parser.add_argument("--db",   default=DEFAULT_DB_DIR, help="Thư mục ChromaDB")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT_DIR, help="Thư mục checkpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--from-checkpoint", action="store_true",
                        help="Load từ checkpoint thay vì re-embed")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Không lưu checkpoint")
    parser.add_argument("-q", "--quiet", action="store_true", help="Tắt verbose log")
    args = parser.parse_args()

    store = run_pipeline(
        csv_path=args.csv,
        db_dir=args.db,
        ckpt_dir=args.ckpt,
        model_name=args.model,
        from_checkpoint=args.from_checkpoint,
        save_ckpt=not args.no_checkpoint,
        verbose=not args.quiet,
    )

    # Quick smoke test sau khi build
    print("\nSmoke test retrieval...")
    from embedder import TrafficLawEmbedder
    emb = TrafficLawEmbedder(model_name=args.model)
    q   = "vượt đèn đỏ bị phạt bao nhiêu tiền?"
    qv  = emb.embed_query(q)

    results = store.search_hybrid(q, qv, top_k=3)
    print(f"\nQuery: '{q}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.4f}] {r.question[:65]}")
