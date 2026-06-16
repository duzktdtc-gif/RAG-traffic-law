"""
evaluator.py
============
Đánh giá chất lượng hệ thống RAG luật giao thông VN.

Các metrics được implement:

  Retrieval Metrics:
    - Hit Rate @K     — có ít nhất 1 relevant doc trong top-K?
    - MRR @K          — Mean Reciprocal Rank
    - NDCG @K         — Normalized Discounted Cumulative Gain
    - Precision @K    — tỉ lệ relevant trong top-K
    - Avg Retrieval Latency

  Answer Quality (nếu có LLM):
    - Faithfulness    — câu trả lời có dựa trên context không?
    - Answer Relevance — câu trả lời có liên quan đến câu hỏi không?
    - Context Utilization — bao nhiêu % context được dùng?

  Ground Truth:
    Dùng lại dataset gốc: với mỗi câu hỏi, câu trả lời đúng = answer trong CSV.
    Relevant doc = doc chứa câu hỏi đó (so khớp doc_id).
    Đây là "closed-set evaluation" — phù hợp để đánh giá retrieval quality.
"""

import sys
import json
import time
import math
import random
import pickle
import re
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from retriever import TrafficLawRetriever, RetrievalResult
from vector_store import SearchResult


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

@dataclass
class EvalSample:
    """Một mẫu đánh giá: query + ground truth."""
    query:          str
    relevant_doc_id: str       # doc_id của document chứa câu trả lời đúng
    ground_truth_answer: str   # câu trả lời đúng (từ dataset)
    domain_tags:    list[str]  = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """Metrics đánh giá retrieval cho một sample."""
    query:         str
    relevant_id:   str
    hit:           bool         # relevant doc có trong top-K?
    rank:          int          # rank của relevant doc (0 = không tìm thấy)
    rr:            float        # Reciprocal Rank (0 nếu không tìm thấy)
    ndcg:          float        # NDCG@K
    precision:     float        # Precision@K
    latency_ms:    float
    n_results:     int


@dataclass
class AggregatedMetrics:
    """Metrics tổng hợp trên toàn bộ eval set."""
    n_samples:      int
    hit_rate:       float       # Hit Rate @K
    mrr:            float       # Mean Reciprocal Rank @K
    ndcg:           float       # NDCG @K (avg)
    precision:      float       # Precision @K (avg)
    avg_latency_ms: float
    k:              int
    search_mode:    str
    per_domain:     dict        = field(default_factory=dict)
    per_intent:     dict        = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_samples":      self.n_samples,
            "k":              self.k,
            "search_mode":    self.search_mode,
            "hit_rate":       round(self.hit_rate, 4),
            "mrr":            round(self.mrr, 4),
            "ndcg":           round(self.ndcg, 4),
            "precision":      round(self.precision, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "per_domain":     self.per_domain,
        }

    def print_report(self):
        print("\n" + "═"*55)
        print(f"  EVALUATION REPORT  (K={self.k}, mode={self.search_mode})")
        print("═"*55)
        print(f"  Samples evaluated : {self.n_samples}")
        print(f"  Hit Rate @{self.k:<2}      : {self.hit_rate:.1%}")
        print(f"  MRR @{self.k:<2}           : {self.mrr:.4f}")
        print(f"  NDCG @{self.k:<2}          : {self.ndcg:.4f}")
        print(f"  Precision @{self.k:<2}     : {self.precision:.4f}")
        print(f"  Avg Latency       : {self.avg_latency_ms:.0f}ms")

        if self.per_domain:
            print(f"\n  Per-Domain Hit Rate @{self.k}:")
            for domain, metrics in sorted(
                self.per_domain.items(),
                key=lambda x: -x[1]["hit_rate"]
            ):
                hr = metrics["hit_rate"]
                n  = metrics["n"]
                bar = "█" * int(hr * 20) + "░" * (20 - int(hr * 20))
                print(f"    {domain:<15} [{bar}] {hr:.1%} (n={n})")
        print("═"*55)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _dcg(relevances: list[float]) -> float:
    """Discounted Cumulative Gain."""
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances)
    )

"""
def _ndcg(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    #NDCG@K với binary relevance (1 relevant doc).
    rels = [1.0 if cid == relevant_id else 0.0 for cid in retrieved_ids[:k]]
    dcg  = _dcg(rels)
    idcg = _dcg([1.0])   # ideal: relevant doc ở rank 1
    return dcg / idcg if idcg > 0 else 0.0
"""


def _ndcg(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """NDCG@K với binary relevance, xử lý chống trùng lặp doc_id."""
    rels = []
    is_counted = False

    # Duyệt qua K kết quả đầu tiên
    for cid in retrieved_ids[:k]:
        # Chỉ tính 1.0 cho lần xuất hiện ĐẦU TIÊN của relevant_id
        if cid == relevant_id and not is_counted:
            rels.append(1.0)
            is_counted = True
        else:
            rels.append(0.0)

    dcg = _dcg(rels)
    idcg = _dcg([1.0])  # ideal: relevant doc ở rank 1

    # Chặn trần 1.0 để tránh lỗi làm tròn số thực (float precision)
    return min(dcg / idcg, 1.0) if idcg > 0 else 0.0


def _find_rank(retrieved: list[SearchResult], relevant_doc_id: str) -> int:
    """Tìm rank của relevant doc (1-indexed, 0 = không tìm thấy)."""
    for i, r in enumerate(retrieved, 1):
        if r.doc_id == relevant_doc_id:
            return i
    return 0


# ─────────────────────────────────────────────
# Eval set builder
# ─────────────────────────────────────────────

def build_eval_set(
    chunks_pkl_path: str,
    n_samples:  int = 200,
    seed:       int = 42,
    strategy:   str = "random",   # "random" | "stratified"
) -> list[EvalSample]:
    """
    Tạo evaluation set từ checkpoint chunks.

    Strategy:
      "random"     — random sample từ toàn bộ
      "stratified" — sample đều theo domain để coverage tốt hơn

    Chỉ lấy các chunk là qa_pair hoặc chunk_index==0 làm query,
    vì đây là các chunk mang câu hỏi rõ ràng nhất.
    """
    with open(chunks_pkl_path, "rb") as f:
        chunks = pickle.load(f)

    # Chỉ lấy "anchor" chunks: chunk_index == 0 (có câu hỏi đầy đủ)
    anchor_chunks = [c for c in chunks if c.metadata.get("chunk_index", 0) == 0]

    random.seed(seed)

    if strategy == "stratified":
        # Group by domain
        by_domain: dict[str, list] = {}
        for c in anchor_chunks:
            tags = c.metadata.get("domain_tags", ["khac"])
            domain = tags[0] if tags else "khac"
            by_domain.setdefault(domain, []).append(c)

        # Sample đều mỗi domain
        per_domain = max(1, n_samples // len(by_domain))
        sampled = []
        for domain, domain_chunks in by_domain.items():
            k = min(per_domain, len(domain_chunks))
            sampled.extend(random.sample(domain_chunks, k))
        sampled = sampled[:n_samples]
    else:
        sampled = random.sample(anchor_chunks, min(n_samples, len(anchor_chunks)))

    eval_samples = []
    for c in sampled:
        # Ground truth answer: lấy từ chunk text (phần sau "Trả lời:")
        text = c.text
        match = re.search(r"Trả lời:\s*", text)
        gt_answer = text[match.end():].strip() if match else text

        eval_samples.append(EvalSample(
            query=c.metadata.get("question", ""),
            relevant_doc_id=c.doc_id,
            ground_truth_answer=gt_answer[:500],
            domain_tags=c.metadata.get("domain_tags", []),
        ))

    # Loại bỏ sample có query rỗng
    eval_samples = [s for s in eval_samples if s.query.strip()]

    return eval_samples


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class RAGEvaluator:
    """
    Đánh giá toàn diện RAG pipeline.

    Usage:
        evaluator = RAGEvaluator(retriever)
        metrics   = evaluator.evaluate(eval_samples, k=5)
        metrics.print_report()
    """

    def __init__(self, retriever: TrafficLawRetriever):
        self.retriever = retriever

    # ── Retrieval evaluation ──────────────────

    def _eval_one(
        self,
        sample: EvalSample,
        k: int,
        search_mode: str,
    ) -> RetrievalMetrics:
        """Đánh giá retrieval cho một sample."""
        t0 = time.time()
        result = self.retriever.retrieve(
            query=sample.query,
            search_mode=search_mode,
            verbose=False,
        )
        latency_ms = (time.time() - t0) * 1000

        # Lấy top-K chunks (retriever có thể trả về ít hơn k)
        retrieved = result.chunks

        rank      = _find_rank(retrieved, sample.relevant_doc_id)
        hit       = rank > 0
        rr        = 1.0 / rank if rank > 0 else 0.0
        retrieved_ids = [r.doc_id for r in retrieved]
        ndcg_val  = _ndcg(retrieved_ids, sample.relevant_doc_id, k)
        precision = sum(1 for r in retrieved[:k] if r.doc_id == sample.relevant_doc_id) / k

        return RetrievalMetrics(
            query=sample.query,
            relevant_id=sample.relevant_doc_id,
            hit=hit,
            rank=rank,
            rr=rr,
            ndcg=ndcg_val,
            precision=precision,
            latency_ms=latency_ms,
            n_results=len(retrieved),
        )

    def evaluate(
        self,
        eval_samples: list[EvalSample],
        k: int = 5,
        search_mode: str = "hybrid",
        verbose: bool = True,
        log_failures: bool = True,
    ) -> AggregatedMetrics:
        """
        Chạy đánh giá trên toàn bộ eval set.

        Args:
            eval_samples:  List EvalSample từ build_eval_set()
            k:             @K cho tất cả metrics
            search_mode:   "hybrid" | "dense" | "bm25"
            verbose:       In progress
            log_failures:  In các query không tìm thấy relevant doc

        Returns:
            AggregatedMetrics với đầy đủ breakdown
        """
        if verbose:
            print(f"\n  [Evaluator] Đánh giá {len(eval_samples)} samples, K={k}, mode={search_mode}")

        per_sample: list[RetrievalMetrics] = []
        failures = []

        for i, sample in enumerate(eval_samples, 1):
            m = self._eval_one(sample, k, search_mode)
            per_sample.append(m)

            if not m.hit:
                failures.append(sample)

            if verbose and i % 20 == 0:
                running_hr = sum(x.hit for x in per_sample) / len(per_sample)
                running_mrr = sum(x.rr for x in per_sample) / len(per_sample)
                print(f"    [{i}/{len(eval_samples)}] "
                      f"Hit Rate: {running_hr:.1%} | MRR: {running_mrr:.4f}")

        # ── Aggregate ─────────────────────────
        n = len(per_sample)
        hit_rate  = sum(m.hit for m in per_sample) / n
        mrr       = sum(m.rr  for m in per_sample) / n
        ndcg      = sum(m.ndcg for m in per_sample) / n
        precision = sum(m.precision for m in per_sample) / n
        avg_lat   = sum(m.latency_ms for m in per_sample) / n

        # ── Per-domain breakdown ──────────────
        domain_hits: dict[str, list[bool]] = {}
        for sample, m in zip(eval_samples, per_sample):
            for tag in (sample.domain_tags or ["khac"]):
                domain_hits.setdefault(tag, []).append(m.hit)

        per_domain = {
            domain: {
                "hit_rate": sum(hits) / len(hits),
                "n": len(hits),
            }
            for domain, hits in domain_hits.items()
        }

        agg = AggregatedMetrics(
            n_samples=n,
            hit_rate=hit_rate,
            mrr=mrr,
            ndcg=ndcg,
            precision=precision,
            avg_latency_ms=avg_lat,
            k=k,
            search_mode=search_mode,
            per_domain=per_domain,
        )

        if log_failures and failures:
            print(f"\n  [Evaluator] ⚠ {len(failures)} queries không tìm thấy relevant doc:")
            for s in failures[:5]:
                print(f"    - {s.query[:70]}")
            if len(failures) > 5:
                print(f"    ... và {len(failures)-5} queries khác")

        return agg

    # ── Compare modes ─────────────────────────

    def compare_search_modes(
        self,
        eval_samples: list[EvalSample],
        k: int = 5,
        modes: list[str] = ["dense", "bm25", "hybrid"],
    ) -> dict[str, AggregatedMetrics]:
        """So sánh các search mode khác nhau."""
        results = {}
        for mode in modes:
            print(f"\n{'─'*40}")
            print(f"  Mode: {mode.upper()}")
            metrics = self.evaluate(eval_samples, k=k, search_mode=mode, verbose=False)
            metrics.print_report()
            results[mode] = metrics
        return results

    # ── Context quality ───────────────────────

    def evaluate_context_quality(
        self,
        eval_samples: list[EvalSample],
        sample_n: int = 20,
    ) -> dict:
        """
        Đánh giá chất lượng context (không cần LLM):
          - Avg context length
          - Avg chunks per query
          - Proportion với references
        """
        sampled = random.sample(eval_samples, min(sample_n, len(eval_samples)))
        stats = {
            "avg_context_chars": [],
            "avg_n_chunks": [],
            "pct_with_refs": [],
        }

        for sample in sampled:
            result = self.retriever.retrieve(sample.query, verbose=False)
            stats["avg_context_chars"].append(len(result.context_text))
            stats["avg_n_chunks"].append(len(result.chunks))
            has_ref = any(c.references for c in result.chunks)
            stats["pct_with_refs"].append(int(has_ref))

        return {
            "avg_context_chars":  round(sum(stats["avg_context_chars"]) / len(sampled)),
            "avg_n_chunks":       round(sum(stats["avg_n_chunks"]) / len(sampled), 2),
            "pct_with_refs":      round(sum(stats["pct_with_refs"]) / len(sampled), 3),
        }


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from embedder import TrafficLawEmbedder
    from vector_store import TrafficLawVectorStore

    CKPT_DIR = "./checkpoints"
    DB_DIR   = "./vectordb"

    print("=== RAG EVALUATOR ===\n")
    print("Loading checkpoint + vectordb...")

    # Load embedder (fit TF-IDF từ corpus)
    with open(f"{CKPT_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(f"{CKPT_DIR}/embeddings.npz")["embeddings"]

    embedder = TrafficLawEmbedder()
    embedder.fit_tfidf([c.text for c in chunks])
    embedder._use_tfidf = True

    # Load vector store
    store = TrafficLawVectorStore(DB_DIR)
    store.load_bm25_from_collection(verbose=True)

    retriever = TrafficLawRetriever(store, embedder, top_k=6, rerank_top_k=5)

    # Build eval set
    print("\nBuilding eval set (stratified, 100 samples)...")
    eval_samples = build_eval_set(
        f"{CKPT_DIR}/chunks.pkl",
        n_samples=100,
        strategy="stratified",
        seed=42,
    )
    print(f"Eval set: {len(eval_samples)} samples")

    # Domain distribution
    from collections import Counter
    domain_dist = Counter()
    for s in eval_samples:
        domain_dist[s.domain_tags[0] if s.domain_tags else "khac"] += 1
    print("Domain distribution:")
    for domain, cnt in domain_dist.most_common():
        print(f"  {domain:<15} {cnt}")

    # Evaluate
    evaluator = RAGEvaluator(retriever)

    print("\n--- Hybrid Search Evaluation ---")
    metrics_hybrid = evaluator.evaluate(
        eval_samples, k=5, search_mode="hybrid", verbose=True, log_failures=True
    )
    metrics_hybrid.print_report()

    # Context quality
    print("\n--- Context Quality ---")
    ctx_quality = evaluator.evaluate_context_quality(eval_samples, sample_n=20)
    print(f"  Avg context length : {ctx_quality['avg_context_chars']:,} chars")
    print(f"  Avg chunks/query   : {ctx_quality['avg_n_chunks']}")
    print(f"  With references    : {ctx_quality['pct_with_refs']:.1%}")

    # Compare modes
    print("\n--- Comparing Search Modes ---")
    comparison = evaluator.compare_search_modes(
        eval_samples[:50], k=5, modes=["dense", "bm25", "hybrid"]
    )

    # Summary table
    print("\n" + "═"*55)
    print("  COMPARISON SUMMARY")
    print("═"*55)
    print(f"  {'Mode':<10} {'Hit@5':>8} {'MRR@5':>8} {'NDCG@5':>8} {'Lat(ms)':>10}")
    print("  " + "─"*51)
    for mode, m in comparison.items():
        print(f"  {mode:<10} {m.hit_rate:>8.1%} {m.mrr:>8.4f} {m.ndcg:>8.4f} {m.avg_latency_ms:>10.0f}")
    print("═"*55)

    # Save results
    results_path = "./checkpoints/eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "hybrid":  metrics_hybrid.to_dict(),
            "context": ctx_quality,
            "comparison": {k: v.to_dict() for k, v in comparison.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Kết quả lưu tại: {results_path}")
