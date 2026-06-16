"""
retriever.py
============
Nhận query → trả về context đã được xử lý, sẵn sàng đưa vào LLM prompt.

Các bước:
  1. Embed query
  2. Hybrid search (Dense + BM25 via RRF)
  3. Post-retrieval filtering (dedup theo doc_id, domain filter)
  4. Cross-encoder reranking (hoặc score-based nếu không có model)
  5. Context packing — ghép chunks thành prompt context với token budget

Output:
    RetrievalResult {
        context_text: str,       # sẵn sàng nhét vào prompt
        chunks: list[SearchResult],
        query_meta: dict,
    }
"""

import time
import re
from dataclasses import dataclass, field
from typing import Optional
from dataclasses import dataclass, field, replace
import numpy as np

from embedder import TrafficLawEmbedder
from vector_store import TrafficLawVectorStore, SearchResult


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_TOP_K        = 6     # số chunks sau hybrid search
RERANK_TOP_K         = 4     # số chunks sau rerank đưa vào prompt
MAX_CONTEXT_CHARS    = 6000  # giới hạn ký tự toàn bộ context
SCORE_THRESHOLD      = 0.01  # loại chunk score quá thấp

# Query expansion: các pattern để nhận diện ý định query
_PENALTY_PATTERN = re.compile(
    r"phạt|mức phạt|xử phạt|tiền phạt|bị phạt|phạt tiền|xử lý",
    re.IGNORECASE,
)
_LAW_CITE_PATTERN = re.compile(
    r"điều\s*\d+|khoản\s*\d+|nghị định|thông tư|luật\s+\w+",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

@dataclass
class RetrievalResult:
    context_text:  str                    # context đã format, đưa thẳng vào prompt
    chunks:        list[SearchResult]     # top-k chunks sau rerank
    query:         str
    retrieval_ms:  float                  # thời gian retrieval (ms)
    query_intent:  list[str]              # ["penalty", "law_cite", ...]
    metadata:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "context_text":  self.context_text,
            "chunks":        [c.to_dict() for c in self.chunks],
            "query":         self.query,
            "retrieval_ms":  self.retrieval_ms,
            "query_intent":  self.query_intent,
            "n_chunks":      len(self.chunks),
            **self.metadata,
        }


# ─────────────────────────────────────────────
# Query analysis
# ─────────────────────────────────────────────

def analyze_query(query: str) -> dict:
    """
    Phân tích ý định query để điều chỉnh retrieval strategy.

    Returns dict với:
        intents: list[str]       — ["penalty", "law_cite", "procedure", ...]
        domain:  str | None      — domain filter gợi ý
        expanded_query: str      — query được expand thêm từ khóa
    """
    intents = []
    q_lower = query.lower()

    if _PENALTY_PATTERN.search(query):
        intents.append("penalty")
    if _LAW_CITE_PATTERN.search(query):
        intents.append("law_cite")
    if any(kw in q_lower for kw in ["thủ tục", "hồ sơ", "đăng ký", "cấp", "gia hạn"]):
        intents.append("procedure")
    if any(kw in q_lower for kw in ["định nghĩa", "là gì", "khái niệm", "được hiểu"]):
        intents.append("definition")
    if any(kw in q_lower for kw in ["điều kiện", "yêu cầu", "phải có", "bắt buộc"]):
        intents.append("condition")

    # Domain inference
    domain = None
    if any(kw in q_lower for kw in ["xe máy", "mô tô", "xe gắn máy"]):
        domain = "xe_may"
    elif any(kw in q_lower for kw in ["ô tô", "xe con", "xe tải"]):
        domain = "oto"
    elif any(kw in q_lower for kw in ["bằng lái", "giấy phép lái xe", "gplx"]):
        domain = "bang_lai"
    elif any(kw in q_lower for kw in ["đường thủy", "tàu thuyền"]):
        domain = "duong_thuy"
    elif any(kw in q_lower for kw in ["đường sắt", "tàu hỏa"]):
        domain = "duong_sat"

    # Query expansion: thêm synonyms
    expanded = query
    expansions = {
        "gplx": "giấy phép lái xe",
        "đk xe": "đăng ký xe",
        "xe ck": "xe chính chủ",
        "cmnd": "căn cước công dân",
        "cccd": "căn cước công dân",
    }
    for abbr, full in expansions.items():
        if abbr in q_lower and full not in q_lower:
            expanded = f"{expanded} {full}"

    return {
        "intents":        intents or ["general"],
        "domain":         domain,
        "expanded_query": expanded,
    }


# ─────────────────────────────────────────────
# Reranker (score-based, no cross-encoder)
# ─────────────────────────────────────────────

def rerank_results(
    results: list[SearchResult],
    query: str,
    query_meta: dict,
    top_k: int = RERANK_TOP_K,
) -> list[SearchResult]:
    """
    Rerank kết quả hybrid search bằng heuristic scoring.
    Khi có neural cross-encoder, thay thế phần này.

    Scoring factors:
      - RRF score (từ hybrid search)
      - Bonus nếu chunk_type == "qa_pair" (câu hỏi khớp trực tiếp)
      - Bonus nếu question của chunk chứa từ khóa query
      - Penalty nếu chunk là continuation (chunk_index > 0) của doc khác
      - Bonus nếu có references (thông tin pháp lý đáng tin cậy hơn)
    """
    query_tokens = set(re.findall(r'\w+', query.lower()))

    scored = []
    for r in results:
        score = r.score

        # Bonus: qa_pair chunk khớp trực tiếp câu hỏi
        if r.chunk_type == "qa_pair":
            score *= 1.15

        # Bonus: question của chunk chứa nhiều từ khóa query
        chunk_q_tokens = set(re.findall(r'\w+', r.question.lower()))
        overlap = len(query_tokens & chunk_q_tokens)
        score += overlap * 0.002

        # Bonus: chunk đầu tiên của doc (chunk_index == 0 có ngữ cảnh đầy đủ)
        if r.chunk_index == 0:
            score *= 1.05

        # Bonus: có references (độ tin cậy cao hơn)
        if r.references:
            score *= 1.03

        # Penalty: intent là "penalty" nhưng chunk không chứa số tiền
        if "penalty" in query_meta.get("intents", []):
            if not re.search(r'\d+[\.,]\d+|\d+\s*(đồng|triệu|nghìn)', r.text, re.IGNORECASE):
                score *= 0.92

        scored.append((score, r))

    scored.sort(key=lambda x: -x[0])

    # Gán lại score đã rerank
    reranked = []
    reranked = [replace(r, score=new_score) for new_score, r in scored[:top_k]]
    
    return reranked


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────

def deduplicate_by_doc(
    results: list[SearchResult],
    max_per_doc: int = 2,
) -> list[SearchResult]:
    """
    Giới hạn tối đa `max_per_doc` chunks từ cùng một document.
    Tránh trường hợp 1 doc chiếm toàn bộ context.
    """
    doc_counts: dict[str, int] = {}
    deduped = []
    for r in results:
        cnt = doc_counts.get(r.doc_id, 0)
        if cnt < max_per_doc:
            deduped.append(r)
            doc_counts[r.doc_id] = cnt + 1
    return deduped


# ─────────────────────────────────────────────
# Context packer
# ─────────────────────────────────────────────

def pack_context(
    chunks: list[SearchResult],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Ghép chunks thành context string cho LLM prompt.

    Format:
        === Tài liệu 1 ===
        Câu hỏi liên quan: ...
        Nội dung: ...
        Nguồn luật: ...

        === Tài liệu 2 ===
        ...
    """
    parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        # Header
        header = f"=== Tài liệu {i} (độ liên quan: {chunk.score:.3f}) ===\n"

        # Question context
        q_line = f"Câu hỏi liên quan: {chunk.question}\n\n" if chunk.question else ""

        # Main text (loại bỏ prefix "Câu hỏi: ..." nếu đã hiển thị)
        text = chunk.text
        # Nếu text bắt đầu bằng "Câu hỏi: ..." thì chỉ lấy phần "Trả lời: ..."
        if text.startswith("Câu hỏi:"):
            match = re.search(r"Trả lời:\s*", text)
            if match:
                text = text[match.end():]

        # References
        ref_lines = ""
        if chunk.references:
            refs = [f"  - {r['text']}" for r in chunk.references[:3] if r.get("text")]
            if refs:
                ref_lines = "\nNguồn luật:\n" + "\n".join(refs)

        # Source link
        src_line = f"\nNguồn: {chunk.source_url}" if chunk.source_url else ""

        block = header + q_line + text + ref_lines + src_line + "\n"

        if total_chars + len(block) > max_chars:
            # Truncate block cuối nếu vượt budget
            remaining = max_chars - total_chars
            if remaining > 200:  # chỉ thêm nếu còn đủ chỗ có ích
                block = block[:remaining] + "...[truncated]"
                parts.append(block)
            break

        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts)


# ─────────────────────────────────────────────
# Main Retriever class
# ─────────────────────────────────────────────

class TrafficLawRetriever:
    """
    End-to-end retriever cho RAG pipeline.

    Usage:
        retriever = TrafficLawRetriever(store, embedder)
        result    = retriever.retrieve("vượt đèn đỏ bị phạt bao nhiêu?")
        # result.context_text → đưa vào LLM prompt
    """

    def __init__(
        self,
        store:    TrafficLawVectorStore,
        embedder: TrafficLawEmbedder,
        top_k:        int = DEFAULT_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        max_per_doc:  int = 2,
    ):
        self.store        = store
        self.embedder     = embedder
        self.top_k        = top_k
        self.rerank_top_k = rerank_top_k
        self.max_per_doc  = max_per_doc

    def retrieve(
        self,
        query: str,
        search_mode: str = "hybrid",   # "dense" | "bm25" | "hybrid"
        domain_filter: Optional[str] = None,
        verbose: bool = False,
    ) -> RetrievalResult:
        """
        Full retrieval pipeline cho một query.

        Args:
            query:         Câu hỏi của user
            search_mode:   "hybrid" (khuyến nghị), "dense", "bm25"
            domain_filter: Filter theo domain tag (vd: "xe_may")
            verbose:       Log chi tiết từng bước

        Returns:
            RetrievalResult với context_text sẵn sàng đưa vào LLM
        """
        t0 = time.time()

        # ── Step 1: Query analysis ─────────────
        query_meta = analyze_query(query)
        expanded_q = query_meta["expanded_query"]

        # Ưu tiên domain filter từ meta nếu không được chỉ định
        if domain_filter is None:
            domain_filter = query_meta.get("domain")

        if verbose:
            print(f"  [Retriever] Query   : {query}")
            print(f"  [Retriever] Expanded: {expanded_q}")
            print(f"  [Retriever] Intents : {query_meta['intents']}")
            print(f"  [Retriever] Domain  : {domain_filter or 'all'}")

        # ── Step 2: Embed query ────────────────
        query_vec = self.embedder.embed_query(expanded_q)

        # ── Step 3: Search ─────────────────────
        if search_mode == "dense":
            raw_results = self.store.search_dense(
                query_vec, self.top_k, filter_domain=domain_filter
            )
        elif search_mode == "bm25":
            raw_results = self.store.search_bm25(expanded_q, self.top_k)
        else:  # hybrid (default)
            raw_results = self.store.search_hybrid(
                expanded_q, query_vec, self.top_k, filter_domain=domain_filter
            )

        if verbose:
            print(f"  [Retriever] Raw results: {len(raw_results)}")

        # ── Step 4: Filter score thấp ──────────
        raw_results = [r for r in raw_results if r.score >= SCORE_THRESHOLD]

        # ── Step 5: Deduplication ──────────────
        deduped = deduplicate_by_doc(raw_results, self.max_per_doc)

        # ── Step 6: Rerank ─────────────────────
        reranked = rerank_results(deduped, query, query_meta, self.rerank_top_k)

        # ── Step 7: Pack context ───────────────
        context_text = pack_context(reranked)

        elapsed_ms = (time.time() - t0) * 1000

        if verbose:
            print(f"  [Retriever] After dedup  : {len(deduped)}")
            print(f"  [Retriever] After rerank : {len(reranked)}")
            print(f"  [Retriever] Context chars: {len(context_text)}")
            print(f"  [Retriever] Time         : {elapsed_ms:.0f}ms")

        return RetrievalResult(
            context_text=context_text,
            chunks=reranked,
            query=query,
            retrieval_ms=elapsed_ms,
            query_intent=query_meta["intents"],
            metadata={
                "search_mode":    search_mode,
                "domain_filter":  domain_filter,
                "n_raw_results":  len(raw_results),
                "n_after_dedup":  len(deduped),
                "expanded_query": expanded_q,
            },
        )


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from indexing_pipeline import run_pipeline

    print("=== RETRIEVER TEST ===\n")
    print("Loading từ checkpoint...")

    # Load từ checkpoint (đã build trước đó)
    from data_loader import load_documents
    from chunker import chunk_all_documents
    from embedder import TrafficLawEmbedder
    from vector_store import TrafficLawVectorStore
    import pickle, numpy as np

    ckpt_dir = "./checkpoints"
    with open(f"{ckpt_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(f"{ckpt_dir}/embeddings.npz")["embeddings"]

    embedder = TrafficLawEmbedder()
    # Fit TF-IDF từ corpus để có thể embed query
    embedder.fit_tfidf([c.text for c in chunks])
    embedder._use_tfidf = True

    store = TrafficLawVectorStore("./vectordb")
    store.load_bm25_from_collection(verbose=True)

    retriever = TrafficLawRetriever(store, embedder)

    test_queries = [
        "vượt đèn đỏ bị phạt bao nhiêu tiền?",
        "không có bằng lái xe máy bị xử phạt như thế nào?",
        "độ tuổi tối thiểu để thi bằng lái xe ô tô hạng B là bao nhiêu?",
        "điều kiện đăng kiểm xe ô tô cũ",
    ]

    for query in test_queries:
        print(f"\n{'═'*60}")
        result = retriever.retrieve(query, verbose=True)
        print(f"\nContext preview (500 chars):")
        print(result.context_text[:500])
        print(f"{'─'*60}")
        print(f"Retrieval time: {result.retrieval_ms:.0f}ms | "
              f"Chunks: {len(result.chunks)} | "
              f"Intent: {result.query_intent}")
