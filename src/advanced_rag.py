"""
advanced_rag.py
===============
Bước 3: Nâng cấp chất lượng retrieval với 3 kỹ thuật:

  1. HyDE (Hypothetical Document Embedding)
     Dùng LLM tạo câu trả lời giả → embed câu đó thay vì embed query gốc.
     Giúp rất nhiều với câu hỏi ngắn ("phạt vượt đèn đỏ?") vì vector
     của câu trả lời giả gần với vector của tài liệu thật hơn.

  2. Cross-Encoder Reranker
     Thay heuristic scoring bằng model thật để rerank top-k chunks.
     Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~85MB, offline).
     Fallback: colbert-style keyword scoring nếu không có model.

  3. Conversation Memory
     Quản lý multi-turn: tự động inject lịch sử hội thoại vào query,
     giải quyết coreference ("thế còn xe máy thì sao?").

Các kỹ thuật này là drop-in replacement — thay thế trực tiếp vào
TrafficLawRetriever mà không cần sửa pipeline khác.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import time
from dataclasses import dataclass, field, replace
import json
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np

from retriever import TrafficLawRetriever, RetrievalResult, analyze_query
from vector_store import SearchResult
from embedder import TrafficLawEmbedder
from generator import TrafficLawGenerator


# ─────────────────────────────────────────────
# 1. HyDE — Hypothetical Document Embedding
# ─────────────────────────────────────────────

HYDE_PROMPT = """Hãy viết một đoạn văn ngắn (3-4 câu) TRẢ LỜI câu hỏi dưới đây về luật giao thông Việt Nam.
Đây là câu trả lời GIẢ ĐỊNH để cải thiện tìm kiếm — hãy viết như thể bạn đang trích dẫn từ văn bản pháp luật.
Bao gồm tên điều luật, nghị định và mức phạt nếu biết.
KHÔNG cần chính xác 100%, chỉ cần viết đúng thể loại văn bản pháp luật.

Câu hỏi: {query}
Câu trả lời giả định:"""


class HyDERetriever:
    """
    Wrapper quanh TrafficLawRetriever thêm HyDE preprocessing.

    Thay vì embed query trực tiếp → dùng LLM tạo hypothetical answer
    → embed câu trả lời giả đó → search.

    Kết quả: recall tốt hơn đặc biệt với câu hỏi ngắn, mơ hồ.
    """

    def __init__(
        self,
        retriever:  TrafficLawRetriever,
        generator:  TrafficLawGenerator,
        embedder:   TrafficLawEmbedder,
        hyde_model: str = "groq_fast",          # dùng model nhanh cho HyDE
        fallback_to_normal: bool = True,        # nếu HyDE fail → dùng retrieval thường
    ):
        self.retriever          = retriever
        self.generator          = generator
        self.embedder           = embedder
        self.hyde_model         = hyde_model
        self.fallback_to_normal = fallback_to_normal

    def _generate_hypothesis(self, query: str) -> str:
        """Dùng LLM tạo câu trả lời giả để embed."""
        try:
            result = self.generator.generate(
                query=query,
                context="",                   # không có context, để LLM tự tạo
                model=self.hyde_model,
            )
            # Chỉ lấy phần đầu nếu quá dài
            hyp = result.answer.strip()[:400]
            return hyp if hyp else query
        except Exception as e:
            print(f"  [HyDE] Lỗi tạo hypothesis: {e}")
            return query

    def retrieve(
        self,
        query: str,
        search_mode: str = "hybrid",
        verbose: bool = False,
    ) -> RetrievalResult:
        """HyDE retrieval: query → hypothesis → embed hypothesis → search."""
        t0 = time.time()

        if verbose:
            print(f"  [HyDE] Generating hypothesis for: {query[:50]}")

        # Bước 1: Tạo hypothetical answer
        hypothesis = _generate_hypothesis_text(query, self.generator, self.hyde_model)

        if verbose:
            print(f"  [HyDE] Hypothesis: {hypothesis[:100]}...")

        # Bước 2: Embed hypothesis thay vì query
        hyp_vec = self.embedder.embed_query(hypothesis)

        # Bước 3: Search với hypothesis vector
        store = self.retriever.store
        if search_mode == "hybrid":
            # Hybrid: hypothesis vector (dense) + query gốc (BM25)
            chunks = store.search_hybrid(
                query=query,               # BM25 vẫn dùng query gốc
                query_vec=hyp_vec,         # Dense dùng hypothesis
                top_k=self.retriever.top_k,
            )
        else:
            chunks = store.search_dense(hyp_vec, top_k=self.retriever.top_k)

        # Bước 4: Đóng gói như RetrievalResult
        from retriever import (
            deduplicate_by_doc, rerank_results,
            pack_context, analyze_query, SCORE_THRESHOLD
        )
        query_meta = analyze_query(query)
        chunks     = [r for r in chunks if r.score >= SCORE_THRESHOLD]
        chunks     = deduplicate_by_doc(chunks, self.retriever.max_per_doc)
        chunks     = rerank_results(chunks, query, query_meta, self.retriever.rerank_top_k)
        context    = pack_context(chunks)

        elapsed_ms = (time.time() - t0) * 1000

        if verbose:
            print(f"  [HyDE] Done: {elapsed_ms:.0f}ms, {len(chunks)} chunks")

        return RetrievalResult(
            context_text=context,
            chunks=chunks,
            query=query,
            retrieval_ms=elapsed_ms,
            query_intent=query_meta["intents"],
            metadata={"search_mode": f"hyde_{search_mode}", "hypothesis": hypothesis[:200]},
        )


def _generate_hypothesis_text(query: str, generator: TrafficLawGenerator, model: str) -> str:
    """Tách ra để dùng độc lập."""
    try:
        # Dùng prompt đơn giản hơn cho LLM fast
        from groq import Groq
        client = Groq(api_key=generator.groq_api_key)
        resp = client.chat.completions.create(
            model=generator.GROQ_FAST,
            temperature=0.3,
            max_tokens=200,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return query


# ─────────────────────────────────────────────
# 2. Cross-Encoder Reranker
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Rerank top-k chunks dùng cross-encoder model.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Size: ~85MB
      - Offline sau khi download
      - Cho điểm relevance(query, passage) chính xác hơn bi-encoder

    Fallback: colbert-style keyword overlap nếu không load được model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name  = model_name
        self._model      = None
        self._use_colbert = False

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            print(f"  [Reranker] Loading {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
            print(f"  [Reranker] Model loaded OK")
        except Exception as e:
            print(f"  [Reranker] Không load được model: {e}, dùng ColBERT fallback")
            self._use_colbert = True

    def _colbert_score(self, query: str, text: str) -> float:
        """Keyword overlap fallback khi không có cross-encoder."""
        q_tokens = set(re.findall(r'\w+', query.lower()))
        t_tokens = set(re.findall(r'\w+', text.lower()))
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & t_tokens) / len(q_tokens)
        # Bonus nếu chứa số tiền (mức phạt)
        if re.search(r'\d+[\.,]\d+|\d+\s*(đồng|triệu|nghìn)', text, re.I):
            overlap *= 1.2
        return min(overlap, 1.0)

    def rerank(
        self,
        query:   str,
        chunks:  list[SearchResult],
        top_k:   int = 4,
    ) -> list[SearchResult]:
        """
        Rerank chunks theo relevance(query, chunk.text).

        Args:
            query:  Câu hỏi gốc
            chunks: List SearchResult từ retriever (đã qua BM25+dense)
            top_k:  Số chunks giữ lại

        Returns:
            List SearchResult đã rerank, score được cập nhật
        """
        if not chunks:
            return chunks

        self._load_model()

        if self._use_colbert or self._model is None:
            # ColBERT fallback
            scored = [(self._colbert_score(query, c.text), c) for c in chunks]
        else:
            # Cross-encoder scoring
            pairs  = [(query, c.text) for c in chunks]
            scores = self._model.predict(pairs)
            scored = list(zip(scores, chunks))

        # Sort và cập nhật score
        scored.sort(key=lambda x: -x[0])
        result = []
        for new_score, chunk in scored[:top_k]:
            result.append(replace(chunk, score=float(new_score)))  # không mutate gốc
        return result


# ─────────────────────────────────────────────
# 3. Conversation Memory
# ─────────────────────────────────────────────

@dataclass
class Turn:
    role:    str    # "user" | "assistant"
    content: str
    query_intent: list[str] = field(default_factory=list)


class ConversationMemory:
    """
    Quản lý lịch sử hội thoại và giải quyết coreference.

    Vấn đề: User hỏi "thế còn xe máy thì sao?" → query này thiếu ngữ cảnh.
    Giải pháp: Rewrite query dựa vào context từ các turn trước.

    Cũng extract "topic" từ cuộc hội thoại để filter domain.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns * 2)

    def add_turn(self, role: str, content: str, intents: list[str] = None):
        self._turns.append(Turn(
            role=role,
            content=content,
            query_intent=intents or [],
        ))

    def get_context_string(self, n_turns: int = 2) -> str:
        """Trả về chuỗi lịch sử để inject vào prompt."""
        turns = list(self._turns)[-n_turns * 2:]
        if not turns:
            return ""
        lines = []
        for t in turns:
            role = "Người dùng" if t.role == "user" else "Trợ lý"
            lines.append(f"{role}: {t.content[:300]}")
        return "Lịch sử hội thoại:\n" + "\n".join(lines)

    def rewrite_query(self, query: str) -> str:
        q_lower = query.lower().strip()

        ambiguous_patterns = [
            r"^(thế|vậy)\s+(còn|thì|nữa|sao)",
            r"^còn\s+\w+\s+(thì sao|bị sao|thì thế nào|thì bị)",
            r"^(nếu|nếu như)\s+\w{1,5}\s+thì",
            r"^(bị phạt|phạt)\s+(thế nào|bao nhiêu|gì)\s*\??$",
            r"^ngoài.{0,20}còn\s+(bị|có)",
        ]

        for p in ambiguous_patterns:
            m = re.search(p, q_lower)

        is_ambiguous = any(re.search(p, q_lower) for p in ambiguous_patterns)

        if not is_ambiguous:
            return query

        user_turns = [t for t in self._turns if t.role == "user"]
        if not user_turns:
            return query

        prev_user_turns = user_turns[:-1]  # bỏ turn hiện tại
        if not prev_user_turns:
            return query
        prev_query = prev_user_turns[-1].content
        rewritten = f"[Ngữ cảnh: {prev_query[:100]}] {query}"
        return rewritten

    def get_topic_domain(self) -> Optional[str]:
        """Infer domain từ toàn bộ cuộc hội thoại."""
        all_intents = []
        for t in self._turns:
            all_intents.extend(t.query_intent)
        if not all_intents:
            return None
        # Domain phổ biến nhất
        from collections import Counter
        c = Counter(all_intents)
        top = c.most_common(1)[0][0]
        return top if top != "general" else None

    def clear(self):
        self._turns.clear()

    def __bool__(self):
        return True

    def __len__(self):
        return len(self._turns)


# ─────────────────────────────────────────────
# Advanced RAG Pipeline (kết hợp cả 3)
# ─────────────────────────────────────────────

class AdvancedRAGPipeline:
    """
    Pipeline nâng cấp kết hợp HyDE + CrossEncoder + ConversationMemory.

    Usage:
        pipeline = AdvancedRAGPipeline(retriever, generator, embedder)
        result   = pipeline.ask("Vượt đèn đỏ bị phạt bao nhiêu?")
        print(result["answer"])
    """

    def __init__(
        self,
        retriever:  TrafficLawRetriever,
        generator:  TrafficLawGenerator,
        embedder:   TrafficLawEmbedder,
        use_hyde:        bool = True,
        use_reranker:    bool = True,
        use_memory:      bool = True,
        reranker_top_k:  int  = 4,
    ):
        self.retriever = retriever
        self.generator = generator
        self.embedder  = embedder

        # Components
        self.hyde     = HyDERetriever(retriever, generator, embedder) if use_hyde else None
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.memory   = ConversationMemory() if use_memory else None

        self.reranker_top_k = reranker_top_k

    def ask(
            self,
            query: str,
            session_id: str = "default",
            search_mode: str = "hybrid",
            model: str = "groq_strong",
            verbose: bool = False,
    ) -> dict:
        t_total = time.time()

        # ── Step 1: Update memory với query hiện tại ──
        if self.memory is not None:
            intents = analyze_query(query)["intents"]
            self.memory.add_turn("user", query, intents)

        # ── Step 2: Query rewrite (memory) ───────
        effective_query = query
        if self.memory is not None:
            effective_query = self.memory.rewrite_query(query)
            if verbose and effective_query != query:
                print(f"  [Memory] Rewritten: {effective_query}")

        # ── Step 3: Retrieve ──────────────────────
        if self.hyde:
            ret = self.hyde.retrieve(effective_query, search_mode, verbose)
        else:
            ret = self.retriever.retrieve(effective_query, search_mode, verbose=verbose)

        # ── Step 4: Rerank ────────────────────────
        chunks = ret.chunks
        if self.reranker and chunks:
            if verbose:
                print(f"  [Reranker] Reranking {len(chunks)} chunks...")
            chunks = self.reranker.rerank(effective_query, chunks, self.reranker_top_k)
            if verbose:
                print(f"  [Reranker] Top score: {chunks[0].score:.4f}")

        # Rebuild context từ reranked chunks
        from retriever import pack_context
        context = pack_context(chunks)

        # ── Step 5: Inject conversation history ──
        if self.memory is not None and len(self.memory) > 1:
            history_ctx = self.memory.get_context_string(n_turns=2)
            context = history_ctx + "\n\n---\n\n" + context

        # ── Step 6: Generate ──────────────────────
        gen = self.generator.generate(
            query=query,
            context=context,
            model=model,
        )

        # ── Step 7: Update memory với answer ──────
        if self.memory is not None:
            self.memory.add_turn("assistant", gen.answer, [])

        total_ms = (time.time() - t_total) * 1000

        return {
            "query": query,
            "effective_query": effective_query,
            "answer": gen.answer,
            "citations": gen.citations,
            "sources": [
                {
                    "question": c.question,
                    "score": round(c.score, 4),
                    "source_url": c.source_url,
                    "chunk_type": c.chunk_type,
                }
                for c in chunks
            ],
            "model_used": gen.model_used,
            "stats": {
                "retrieval_ms": round(ret.retrieval_ms, 1),
                "generate_ms": round(gen.latency_ms, 1),
                "total_ms": round(total_ms, 1),
                "input_tokens": gen.input_tokens,
                "output_tokens": gen.output_tokens,
                "hyde_used": self.hyde is not None,
                "reranker_used": self.reranker is not None,
            },
        }
    def clear_memory(self):
        if self.memory:
            self.memory.clear()


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys, pickle
    sys.path.insert(0, "src")
    from embedder import TrafficLawEmbedder
    from vector_store import TrafficLawVectorStore
    from retriever import TrafficLawRetriever
    from generator import TrafficLawGenerator

    print("=== ADVANCED RAG TEST ===\n")

    # Load pipeline
    with open("checkpoints/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = TrafficLawEmbedder()
    embedder.fit_tfidf([c.text for c in chunks])
    embedder._use_tfidf = True

    store = TrafficLawVectorStore("./vectordb")
    store.load_bm25_from_collection(verbose=False)
    retriever  = TrafficLawRetriever(store, embedder)
    generator  = TrafficLawGenerator(groq_api_key=os.getenv("GROQ_API_KEY"))

    pipeline = AdvancedRAGPipeline(
        retriever=retriever,
        generator=generator,
        embedder=embedder,
        use_hyde=False,         # Tắt HyDE trong test để tiết kiệm quota
        use_reranker=True,      # Test reranker
        use_memory=True,        # Test memory
    )

    # Simulate multi-turn conversation
    conversation = [
        "Vượt đèn đỏ xe máy bị phạt bao nhiêu tiền?",
        "Thế còn ô tô thì sao?",                          # ambiguous → memory rewrite
        "Ngoài phạt tiền thì còn bị gì nữa không?",       # follow-up
    ]

    for query in conversation:
        print(f"\n{'═'*60}")
        print(f"Q: {query}")
        result = pipeline.ask(query, model="groq_fast", verbose=True)
        print(f"\nEffective query: {result['effective_query']}")
        print(f"Model: {result['model_used']} | "
              f"Total: {result['stats']['total_ms']:.0f}ms")
        print(f"\nAnswer:\n{result['answer'][:500]}")
        if result['citations']:
            print(f"\nCitations: {result['citations'][:3]}")
