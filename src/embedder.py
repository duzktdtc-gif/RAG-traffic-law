"""
embedder.py
===========
Tạo vector embedding cho Chunk dùng SentenceTransformer.

Model được chọn: intfloat/multilingual-e5-small
  - Hỗ trợ tiếng Việt tốt (trained on 100+ languages)
  - 384-dim vectors, nhỏ gọn (~120MB)
  - Phù hợp với instruction-based retrieval:
      query  → "query: <text>"
      doc    → "passage: <text>"

Thiết kế:
  - Batch embedding với progress bar
  - Cache model để không load lại nhiều lần
  - Normalize vectors để dùng cosine similarity
  - Fallback sang TF-IDF nếu model không load được
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

# Tắt logging của transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"

from chunker import Chunk


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_MODEL    = "intfloat/multilingual-e5-small"
FALLBACK_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384
DEFAULT_BATCH    = 64

# Instruction prefix theo chuẩn E5
QUERY_PREFIX     = "query: "
PASSAGE_PREFIX   = "passage: "


# ─────────────────────────────────────────────
# Embedder class
# ─────────────────────────────────────────────

class TrafficLawEmbedder:
    """
    Wrapper quanh SentenceTransformer với:
      - Lazy loading (chỉ load khi gọi lần đầu)
      - Instruction prefix cho query vs passage
      - Normalize L2 tự động
      - TF-IDF fallback nếu model không khả dụng
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device     = device
        self.cache_dir  = cache_dir
        self._model     = None
        self._use_tfidf = False
        self._tfidf_vec     = None   # fitted flag (True khi đã fit)
        self._tfidf_tfidf   = None
        self._tfidf_svd     = None
        self._tfidf_dim     = EMBED_DIM

    # ── Private ───────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        # Nếu đã switch sang TF-IDF (và đã fit), không cần load neural model nữa
        if self._use_tfidf and self._tfidf_vec is True:
            return
        try:
            print(f"  [Embedder] Loading model: {self.model_name} ...")
            t0 = time.time()
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            print(f"  [Embedder] Model loaded trong {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  [Embedder] Không load được {self.model_name}: {e}")
            # Thử fallback model
            try:
                print(f"  [Embedder] Thử fallback: {FALLBACK_MODEL}")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    FALLBACK_MODEL,
                    device=self.device,
                    cache_folder=self.cache_dir,
                )
                self.model_name = FALLBACK_MODEL
                print(f"  [Embedder] Fallback OK")
            except Exception as e2:
                print(f"  [Embedder] ⚠ Dùng TF-IDF fallback: {e2}")
                self._use_tfidf = True

    def _tfidf_fit(self, texts: list[str]):
        """
        Fit TF-IDF + SVD trên corpus.
        n_components = min(EMBED_DIM, n_features-1) để tránh lỗi SVD.
        Nếu corpus quá nhỏ (<2 docs), dùng raw TF-IDF vector (không SVD).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        print("  [Embedder] Fit TF-IDF (char 2-4gram, max 50k features)...")
        self._tfidf_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50_000,
            sublinear_tf=True,
        )
        X = self._tfidf_tfidf.fit_transform(texts)
        n_comp = min(EMBED_DIM, X.shape[0] - 1, X.shape[1] - 1)
        print(f"  [Embedder] SVD: {X.shape} -> {n_comp} dims")

        if n_comp < 1:
            # Corpus quá nhỏ: dùng raw TF-IDF + zero-pad
            self._tfidf_svd  = None
            self._tfidf_dim  = min(EMBED_DIM, X.shape[1])
        else:
            self._tfidf_svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self._tfidf_svd.fit(X)
            self._tfidf_dim  = n_comp
        self._tfidf_vec  = True   # fitted flag

    def _tfidf_transform(self, texts: list[str]) -> np.ndarray:
        X = self._tfidf_tfidf.transform(texts)
        if self._tfidf_svd is not None:
            vecs = self._tfidf_svd.transform(X).astype(np.float32)
        else:
            # Không có SVD (corpus quá nhỏ lúc fit): dùng raw TF-IDF, truncate/pad
            vecs = X.toarray().astype(np.float32)
            if vecs.shape[1] > EMBED_DIM:
                vecs = vecs[:, :EMBED_DIM]
        # Pad to EMBED_DIM if needed
        if vecs.shape[1] < EMBED_DIM:
            pad  = np.zeros((vecs.shape[0], EMBED_DIM - vecs.shape[1]), dtype=np.float32)
            vecs = np.hstack([vecs, pad])
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def _tfidf_embed(self, texts: list[str]) -> np.ndarray:
        """TF-IDF + SVD fallback cho offline/resource-constrained environment."""
        if self._tfidf_vec is None:
            self._tfidf_fit(texts)
        return self._tfidf_transform(texts)

    def fit_tfidf(self, corpus_texts: list[str]):
        """
        Pre-fit TF-IDF trên toàn bộ corpus TRƯỚC khi embed.
        Gọi phương thức này khi dùng TF-IDF fallback để đảm bảo
        vocabulary đầy đủ, tránh out-of-vocab khi embed query.
        """
        self._tfidf_fit(corpus_texts)

    def _neural_embed(
        self,
        texts: list[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2 normalize
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    # ── Public API ────────────────────────────

    @property
    def embed_dim(self) -> int:
        return EMBED_DIM

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed danh sách văn bản (documents/passages).
        Tự động thêm prefix 'passage: ' cho E5 model.

        Returns:
            np.ndarray shape (N, embed_dim), dtype float32, L2-normalized
        """
        self._load_model()
        if self._use_tfidf:
            return self._tfidf_embed(texts)

        prefixed = [PASSAGE_PREFIX + t for t in texts]
        return self._neural_embed(prefixed, batch_size, show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed một câu query (single vector).
        Tự động thêm prefix 'query: ' cho E5 model.

        Returns:
            np.ndarray shape (embed_dim,), L2-normalized
        """
        self._load_model()
        if self._use_tfidf:
            return self._tfidf_embed([query])[0]

        prefixed = QUERY_PREFIX + query
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = DEFAULT_BATCH,
        verbose: bool = True,
    ) -> tuple[list[Chunk], np.ndarray]:
        """
        Embed toàn bộ Chunk list.

        Returns:
            (chunks, embeddings) — embeddings shape (N, embed_dim)
        """
        texts = [c.text for c in chunks]
        t0 = time.time()

        if verbose:
            print(f"\n  [Embedder] Embedding {len(chunks):,} chunks (batch={batch_size})...")

        vecs = self.embed_passages(texts, batch_size, show_progress=verbose)
        elapsed = time.time() - t0

        if verbose:
            speed = len(chunks) / elapsed
            print(f"  [Embedder] Done: {elapsed:.1f}s · {speed:.0f} chunks/s")
            print(f"  [Embedder] Matrix shape : {vecs.shape}")
            print(f"  [Embedder] Model        : {self.model_name if not self._use_tfidf else 'TF-IDF+SVD'}")
            # Kiểm tra norm
            norms = np.linalg.norm(vecs, axis=1)
            print(f"  [Embedder] Norm avg/min : {norms.mean():.4f} / {norms.min():.4f}")

        return chunks, vecs


# ─────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────

_default_embedder: Optional[TrafficLawEmbedder] = None


def get_embedder(
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[str] = None,
) -> TrafficLawEmbedder:
    """Trả về embedder singleton (lazy init)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = TrafficLawEmbedder(model_name=model_name, cache_dir=cache_dir)
    return _default_embedder


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, "src")
    from data_loader import load_documents
    from chunker import chunk_all_documents

    docs   = load_documents("/mnt/user-data/uploads/Dataset_6000.csv", verbose=False)
    chunks = chunk_all_documents(docs[:50], verbose=False)

    embedder = TrafficLawEmbedder()
    _, vecs  = embedder.embed_chunks(chunks[:30], verbose=True)

    # Test query embedding
    query = "vượt đèn đỏ bị phạt bao nhiêu tiền?"
    qvec  = embedder.embed_query(query)
    sims  = vecs @ qvec

    print(f"\n  Query: '{query}'")
    print(f"  Top-3 matches:")
    top3 = np.argsort(sims)[::-1][:3]
    for i in top3:
        print(f"    [{sims[i]:.4f}] {chunks[i].metadata['question'][:70]}")
