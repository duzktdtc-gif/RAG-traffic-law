"""
data_loader.py
==============
Đọc, validate và chuẩn hóa Dataset_6000.csv thành dạng Document chuẩn
cho pipeline RAG luật giao thông Việt Nam.

Document schema:
    {
        "doc_id":    str,          # unique ID
        "question":  str,          # câu hỏi gốc
        "answer":    str,          # câu trả lời gốc (raw)
        "references": list[dict],  # [{text, url}]
        "source_url": str,         # link trang gốc
        "metadata":  dict          # domain tags, stats…
    }
"""

import ast
import hashlib
import re
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

@dataclass
class Reference:
    text: str
    url: str


@dataclass
class Document:
    doc_id:     str
    question:   str
    answer:     str
    references: list[Reference] = field(default_factory=list)
    source_url: str = ""
    metadata:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["references"] = [asdict(r) for r in self.references]
        return d


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_doc_id(question: str, idx: int) -> str:
    """SHA-8 của câu hỏi + index để tránh collision khi duplicate."""
    h = hashlib.sha1(question.encode()).hexdigest()[:8]
    return f"doc_{idx:05d}_{h}"


def _parse_references(raw: str) -> list[Reference]:
    """
    References trong CSV có dạng Python-literal list of tuples:
        [('Điều 56 Luật ...', 'https://...'), ...]
    Hoặc rỗng [] / NaN.
    """
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw)
        refs = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                refs.append(Reference(text=str(item[0]).strip(),
                                      url=str(item[1]).strip()))
            elif isinstance(item, str):
                refs.append(Reference(text=item.strip(), url=""))
        return refs
    except Exception:
        return []


_DOMAIN_PATTERNS = {
    "xe_may":      [r"xe\s*(máy|gắn máy|mô\s*tô)", r"mô\s*tô"],
    "oto":         [r"ô\s*tô", r"xe\s*con", r"xe\s*tải", r"xe\s*buýt"],
    "bang_lai":    [r"giấy phép lái xe", r"bằng lái", r"gplx"],
    "xu_phat":     [r"xử phạt", r"phạt tiền", r"vi phạm", r"phạt"],
    "dang_ky_xe":  [r"đăng ký xe", r"giấy đăng ký", r"biển số"],
    "bao_hiem":    [r"bảo hiểm"],
    "ruou_bia":    [r"rượu", r"bia", r"nồng độ cồn"],
    "toc_do":      [r"tốc độ", r"vượt tốc", r"quá tốc"],
    "den_tin_hieu":[r"đèn đỏ", r"đèn tín hiệu", r"vượt đèn"],
    "dang_kiem":   [r"đăng kiểm", r"kiểm định"],
    "duong_bo":    [r"đường bộ", r"đường cao tốc", r"vỉa hè"],
    "duong_thuy":  [r"đường thủy", r"thủy nội địa", r"tàu thuyền"],
    "hang_khong":  [r"hàng không", r"sân bay", r"cảng hàng không"],
    "duong_sat":   [r"đường sắt", r"tàu hỏa", r"đường ray"],
}


def _tag_domain(text: str) -> list[str]:
    """Gán domain tags cho document dựa vào nội dung."""
    text_lower = text.lower()
    tags = []
    for tag, patterns in _DOMAIN_PATTERNS.items():
        if any(re.search(p, text_lower) for p in patterns):
            tags.append(tag)
    return tags or ["khac"]


# ─────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────

def load_documents(
    csv_path: str | Path,
    drop_duplicates: bool = True,
    min_answer_len: int = 50,
    verbose: bool = True,
) -> list[Document]:
    """
    Đọc CSV và trả về list[Document] đã validate + enrich metadata.

    Args:
        csv_path:        Đường dẫn tới Dataset_6000.csv
        drop_duplicates: Bỏ câu hỏi trùng lặp (giữ lần đầu)
        min_answer_len:  Lọc answer quá ngắn (< N ký tự)
        verbose:         In thống kê ra stdout

    Returns:
        List các Document hợp lệ, sẵn sàng đưa vào chunker.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ── Validate columns ──────────────────────────────────
    required = {"question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    raw_count = len(df)

    # ── Drop NaN ──────────────────────────────────────────
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].str.strip()
    df["answer"]   = df["answer"].str.strip()

    # ── Drop quá ngắn ─────────────────────────────────────
    df = df[df["answer"].str.len() >= min_answer_len]

    # ── Drop duplicates ───────────────────────────────────
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["question"], keep="first")
        dupes_dropped = before - len(df)
    else:
        dupes_dropped = 0

    df = df.reset_index(drop=True)

    # ── Build Document list ───────────────────────────────
    docs: list[Document] = []
    for i, row in df.iterrows():
        question   = row["question"]
        answer     = row["answer"]
        source_url = row.get("request_link", "") or ""
        refs_raw   = row.get("references", "") or ""

        refs    = _parse_references(str(refs_raw))
        tags    = _tag_domain(question + " " + answer)
        doc_id  = _make_doc_id(question, i)

        metadata = {
            "domain_tags":  tags,
            "answer_len":   len(answer),
            "question_len": len(question),
            "has_refs":     len(refs) > 0,
            "ref_count":    len(refs),
        }

        docs.append(Document(
            doc_id=doc_id,
            question=question,
            answer=answer,
            references=refs,
            source_url=str(source_url).strip(),
            metadata=metadata,
        ))

    # ── Stats ─────────────────────────────────────────────
    if verbose:
        ans_lens = [d.metadata["answer_len"] for d in docs]
        tag_counts: dict[str, int] = {}
        for d in docs:
            for t in d.metadata["domain_tags"]:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        print("=" * 55)
        print("  DATA LOADER — Thống kê")
        print("=" * 55)
        print(f"  Raw rows        : {raw_count:,}")
        print(f"  Sau khi drop NaN: {raw_count - (raw_count - len(df) - dupes_dropped):,}")
        print(f"  Duplicates bỏ  : {dupes_dropped:,}")
        print(f"  Documents hợp lệ: {len(docs):,}")
        print(f"  Answer len avg  : {sum(ans_lens)/len(ans_lens):,.0f} ký tự")
        print(f"  Answer len min  : {min(ans_lens):,} ký tự")
        print(f"  Answer len max  : {max(ans_lens):,} ký tự")
        print(f"  Có references   : {sum(d.metadata['has_refs'] for d in docs):,}")
        print()
        print("  Domain tags (top 10):")
        for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * (cnt * 30 // max(tag_counts.values()))
            print(f"    {tag:<15} {cnt:>5,}  {bar}")
        print("=" * 55)

    return docs


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Dataset_6000.csv"
    docs = load_documents(csv)
    print(f"\nSample document:")
    d = docs[3]
    print(f"  ID       : {d.doc_id}")
    print(f"  Question : {d.question[:80]}")
    print(f"  Answer   : {d.answer[:120]}...")
    print(f"  Tags     : {d.metadata['domain_tags']}")
    print(f"  Refs     : {len(d.references)}")
