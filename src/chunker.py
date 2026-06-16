"""
chunker.py
==========
Chia Document thành Chunk để embedding.

Chiến lược chính (theo đặc điểm dataset luật giao thông VN):

  1. QA-Pair chunk  — ghép question + answer thành 1 chunk ngắn.
                      Tốt cho câu hỏi trực tiếp.

  2. Sliding-Window — với answer dài (>= LONG_ANSWER_CHARS),
                      cắt thành các cửa sổ chồng lấp để giữ ngữ cảnh.

  3. Article-Split  — với answer chứa cấu trúc điều/khoản (Điều X, Khoản Y…)
                      tách theo ranh giới điều luật.

Mỗi Chunk mang đủ metadata để truy vết ngược về Document gốc
và để filter trong vector store.

Chunk schema:
    {
        "chunk_id":    str,
        "doc_id":      str,
        "chunk_type":  "qa_pair" | "sliding_window" | "article_split",
        "text":        str,          # văn bản sẽ được embed
        "token_count": int,
        "metadata":    dict          # kế thừa từ Document + chunk-level
    }
"""

import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Literal

from data_loader import Document


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# Token estimation: tiếng Việt trung bình ~3.5 ký tự/token
# (thực tế đo trên BPE tokenizers với văn bản pháp luật VN)
_CHARS_PER_TOKEN = 3.5

MAX_CHUNK_TOKENS   = 512    # giới hạn tokens mỗi chunk (~1,792 ký tự)
OVERLAP_TOKENS     = 64     # overlap khi sliding window  (~224 ký tự)
LONG_ANSWER_CHARS  = 800    # ngưỡng dùng sliding window thay QA-pair
ARTICLE_PATTERN    = re.compile(
    r"(?=(?:Điều|Khoản|Mục|Chương|Điểm)\s+\d+[\w\s]*[:\.]\s)",
    re.IGNORECASE,
)
CHUNK_HEADER_TEMPLATE = "Câu hỏi: {q}\n\nTrả lời: {a}"


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

ChunkType = Literal["qa_pair", "sliding_window", "article_split"]


@dataclass
class Chunk:
    chunk_id:   str
    doc_id:     str
    chunk_type: ChunkType
    text:       str
    token_count: int
    metadata:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Ước tính số tokens. Tiếng Việt ~3.5 ký tự/token."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _make_chunk_id(doc_id: str, idx: int, chunk_type: str) -> str:
    raw = f"{doc_id}_{chunk_type}_{idx}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"chunk_{h}"


def _clean_text(text: str) -> str:
    """Bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _sliding_window_chunks(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """
    Cắt văn bản thành các cửa sổ token chồng lấp.
    Cắt tại ranh giới câu (dấu . ! ?) để tránh vỡ giữa chừng.
    """
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    chunks, current, current_tokens = [], [], 0

    for sent in sentences:
        t = _count_tokens(sent)
        if current_tokens + t > max_tokens and current:
            chunks.append(" ".join(current))
            # Giữ lại overlap từ cuối
            overlap_buf, overlap_tok = [], 0
            for s in reversed(current):
                st = _count_tokens(s)
                if overlap_tok + st <= overlap_tokens:
                    overlap_buf.insert(0, s)
                    overlap_tok += st
                else:
                    break
            current, current_tokens = overlap_buf, overlap_tok

        current.append(sent)
        current_tokens += t

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def _article_split_chunks(answer: str) -> list[str]:
    """
    Tách answer theo ranh giới điều luật (Điều X, Khoản Y…).
    Trả về danh sách đoạn, mỗi đoạn không vượt MAX_CHUNK_TOKENS.
    """
    parts = ARTICLE_PATTERN.split(answer)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _count_tokens(part) > MAX_CHUNK_TOKENS:
            result.extend(_sliding_window_chunks(part))
        else:
            result.append(part)
    return result


def _has_article_structure(text: str) -> bool:
    """Kiểm tra answer có chứa điều/khoản dạng luật không."""
    return bool(ARTICLE_PATTERN. findall(text)) >= 2


# ─────────────────────────────────────────────
# Core: chunk một Document
# ─────────────────────────────────────────────

def chunk_document(doc: Document) -> list[Chunk]:
    """
    Áp dụng chiến lược chunking phù hợp với độ dài và cấu trúc.

    Logic:
        answer ngắn (<= LONG_ANSWER_CHARS)   → QA-Pair chunk (1 chunk)
        answer có cấu trúc điều/khoản         → Article-Split
        answer dài, không có điều/khoản       → Sliding-Window
    """
    answer  = _clean_text(doc.answer)
    question = doc.question.strip()

    # Metadata kế thừa từ document
    base_meta = {
        "doc_id":       doc.doc_id,
        "source_url":   doc.source_url,
        "domain_tags":  doc.metadata.get("domain_tags", []),
        "has_refs":     doc.metadata.get("has_refs", False),
        "question":     question,                    # để filter / hybrid search
        "references":   [{"text": r.text, "url": r.url} for r in doc.references],
    }

    chunks: list[Chunk] = []

    # ── Strategy A: QA-Pair (answer ngắn) ────────────────
    if len(answer) <= LONG_ANSWER_CHARS:
        text = CHUNK_HEADER_TEMPLATE.format(q=question, a=answer)
        tok  = _count_tokens(text)
        cid  = _make_chunk_id(doc.doc_id, 0, "qa_pair")
        chunks.append(Chunk(
            chunk_id=cid,
            doc_id=doc.doc_id,
            chunk_type="qa_pair",
            text=text,
            token_count=tok,
            metadata={**base_meta, "chunk_index": 0, "total_chunks": 1},
        ))
        return chunks

    # ── Strategy B: Article-Split (có cấu trúc điều luật) ─
    if _has_article_structure(answer):
        parts = _article_split_chunks(answer)
        for i, part in enumerate(parts):
            # Luôn prefix câu hỏi vào chunk đầu tiên để giữ ngữ cảnh
            if i == 0:
                text = CHUNK_HEADER_TEMPLATE.format(q=question, a=part)
            else:
                text = f"[Tiếp theo] {part}"
            tok = _count_tokens(text)
            cid = _make_chunk_id(doc.doc_id, i, "article_split")
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc.doc_id,
                chunk_type="article_split",
                text=text,
                token_count=tok,
                metadata={**base_meta, "chunk_index": i, "total_chunks": len(parts)},
            ))
        return chunks

    # ── Strategy C: Sliding-Window (answer dài, không cấu trúc) ─
    windows = _sliding_window_chunks(answer)
    for i, window in enumerate(windows):
        if i == 0:
            text = CHUNK_HEADER_TEMPLATE.format(q=question, a=window)
        else:
            text = f"[Câu hỏi: {question}]\n\n{window}"
        tok = _count_tokens(text)
        cid = _make_chunk_id(doc.doc_id, i, "sliding_window")
        chunks.append(Chunk(
            chunk_id=cid,
            doc_id=doc.doc_id,
            chunk_type="sliding_window",
            text=text,
            token_count=tok,
            metadata={**base_meta, "chunk_index": i, "total_chunks": len(windows)},
        ))
    return chunks


# ─────────────────────────────────────────────
# Batch chunker
# ─────────────────────────────────────────────

def chunk_all_documents(
    docs: list[Document],
    verbose: bool = True,
) -> list[Chunk]:
    """Chunk toàn bộ document list, trả về flat list chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    if verbose:
        type_counts: dict[str, int] = {}
        token_list = []
        for c in all_chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
            token_list.append(c.token_count)

        avg_tok = sum(token_list) / len(token_list)
        print("=" * 55)
        print("  CHUNKER — Thống kê")
        print("=" * 55)
        print(f"  Documents đầu vào : {len(docs):,}")
        print(f"  Chunks tạo ra     : {len(all_chunks):,}")
        print(f"  Avg chunks/doc    : {len(all_chunks)/len(docs):.2f}")
        print(f"  Token avg/chunk   : {avg_tok:.0f}")
        print(f"  Token min/max     : {min(token_list)} / {max(token_list)}")
        print()
        print("  Chunk type breakdown:")
        for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = cnt * 100 / len(all_chunks)
            bar = "█" * int(pct / 2)
            print(f"    {t:<20} {cnt:>6,}  ({pct:.1f}%)  {bar}")
        print("=" * 55)

    return all_chunks


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_documents

    docs = load_documents("/mnt/user-data/uploads/Dataset_6000.csv", verbose=False)
    chunks = chunk_all_documents(docs[:200], verbose=True)

    print("\nSample chunks:")
    for ct in ["qa_pair", "article_split", "sliding_window"]:
        sample = next((c for c in chunks if c.chunk_type == ct), None)
        if sample:
            print(f"\n[{ct}] chunk_id={sample.chunk_id}")
            print(f"  tokens={sample.token_count}")
            print(f"  text preview: {sample.text[:180]}...")
