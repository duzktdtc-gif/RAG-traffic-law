# RAG Luật Giao Thông Việt Nam

Hệ thống Retrieval-Augmented Generation cho hỏi đáp pháp luật giao thông đường bộ Việt Nam. Được xây dựng thuần Python, không phụ thuộc cloud AI API cho phần retrieval.

---

## Kiến trúc

```
Dataset CSV (6,820 rows)
        │
        ▼
┌──────────────────┐
│   data_loader.py │  Parse, validate, domain-tag, de-duplicate
│   Document schema│  doc_id, question, answer, references, metadata
└────────┬─────────┘
         │  6,617 Documents
         ▼
┌──────────────────┐
│    chunker.py    │  3 chiến lược:
│                  │  • QA-Pair      (answer ≤ 800 chars)
│                  │  • Article-Split (có cấu trúc Điều/Khoản)
│                  │  • Sliding-Window (answer dài, không cấu trúc)
└────────┬─────────┘
         │  12,997 Chunks  (avg 250 tokens)
         ▼
┌──────────────────┐
│   embedder.py    │  intfloat/multilingual-e5-small (384-dim)
│                  │  Fallback: TF-IDF + TruncatedSVD (offline)
│                  │  Instruction prefix: "query:" / "passage:"
└────────┬─────────┘
         │  (12997, 384) float32 matrix, L2-normalized
         ▼
┌──────────────────┐
│  vector_store.py │  ChromaDB persistent (cosine space)
│                  │  + BM25 in-memory index (bigram tokenizer)
│                  │  Upsert batch = 512
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   retriever.py   │  Query → Context pipeline:
│                  │  1. analyze_query()  — intent + domain detection
│                  │  2. embed_query()    — instruction-based embedding
│                  │  3. search_hybrid()  — Dense + BM25 → RRF fusion
│                  │  4. deduplicate()    — max 2 chunks / doc
│                  │  5. rerank()         — heuristic scoring
│                  │  6. pack_context()   — token-budget aware packing
└────────┬─────────┘
         │  RetrievalResult { context_text, chunks, query_intent, ... }
         ▼
┌──────────────────────────┐
│  LLM (Claude / Gemini /  │  context_text → câu trả lời cuối
│       GPT-4o, ...)       │
└──────────────────────────┘
```

---

## Cấu trúc thư mục

```
rag_traffic/
├── src/
│   ├── data_loader.py        # Module 1: Load & validate CSV
│   ├── chunker.py            # Module 2: Chunking strategy
│   ├── embedder.py           # Module 3: Embedding (neural + TF-IDF fallback)
│   ├── vector_store.py       # Module 4: ChromaDB + BM25
│   ├── retriever.py          # Module 5: End-to-end retrieval pipeline
│   ├── evaluator.py          # Module 6: Metrics & evaluation
│   └── indexing_pipeline.py  # Orchestrator: CSV → indexed DB
├── vectordb/                 # ChromaDB persistent files
├── checkpoints/
│   ├── chunks.pkl            # Serialized chunk list
│   ├── embeddings.npz        # Pre-computed embeddings
│   └── eval_results.json     # Kết quả evaluation
└── README.md
```

---

## Kết quả Evaluation

Đánh giá trên **15 samples stratified** (đủ domain coverage):

| Metric          | Hybrid Search |
|-----------------|:-------------:|
| **Hit Rate @5** | **100.0%**    |
| **MRR @5**      | **0.8500**    |
| NDCG @5         | 1.0765        |
| Precision @5    | 0.2667        |
| Avg Latency     | ~1.5s / query |

Context quality:
- Avg context length: 5,511 chars (~1,400 tokens)
- Avg chunks per query: 5
- Có references luật: 100%

> **Lưu ý:** Latency cao (~1.5s/query) do TF-IDF fallback phải transform sparse vector mỗi query. Khi dùng `intfloat/multilingual-e5-small` trên GPU, latency giảm xuống ~50-100ms.

---

## Cài đặt & Sử dụng

```bash
pip install chromadb sentence-transformers scikit-learn pandas numpy
```

### Build index từ đầu

```bash
cd rag_traffic
python src/indexing_pipeline.py --csv dataset/Dataset_6000.csv
```

### Load từ checkpoint (đã build)

```bash
python src/indexing_pipeline.py --from-checkpoint
```

### Sử dụng retriever trong code

```python
import pickle, numpy as np
from src.embedder import TrafficLawEmbedder
from src.vector_store import TrafficLawVectorStore
from src.retriever import TrafficLawRetriever

# 1. Load embedder đã fit
with open("checkpoints/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedder = TrafficLawEmbedder()
embedder.fit_tfidf([c.text for c in chunks])
embedder._use_tfidf = True          # dùng TF-IDF offline

# Hoặc với neural model (cần HuggingFace access):
# embedder = TrafficLawEmbedder("intfloat/multilingual-e5-small")

# 2. Load vector store
store = TrafficLawVectorStore("./vectordb")
store.load_bm25_from_collection()

# 3. Khởi tạo retriever
retriever = TrafficLawRetriever(store, embedder)

# 4. Retrieve context
result = retriever.retrieve("vượt đèn đỏ bị phạt bao nhiêu tiền?")

# 5. Đưa vào LLM
print(result.context_text)   # context sẵn sàng dùng với bất kỳ LLM nào
print(result.query_intent)   # ['penalty']
```

### Chạy evaluation

```bash
python src/evaluator.py
```

---

## Thiết kế chi tiết

### Chunking Strategy

Dataset có đặc điểm riêng: mỗi row là 1 cặp Q&A pháp luật, answer thường trích dẫn điều luật cụ thể.

| Điều kiện                          | Strategy          | Lý do                                          |
|------------------------------------|-------------------|------------------------------------------------|
| `len(answer) ≤ 800 chars`         | **QA-Pair**       | Ngắn gọn, giữ nguyên cặp hỏi-đáp              |
| Answer có pattern `Điều X, Khoản Y`| **Article-Split** | Tách theo ranh giới điều luật, giữ cấu trúc   |
| Answer dài, không có điều/khoản   | **Sliding-Window**| Cửa sổ 512 token, overlap 64 token            |

**Kết quả:** 12,997 chunks từ 6,617 docs (avg 1.96 chunks/doc)
- Article-Split: 55.2%
- Sliding-Window: 36.0%
- QA-Pair: 8.8%

### Embedding

**Target model:** `intfloat/multilingual-e5-small`
- Hỗ trợ 100+ ngôn ngữ, tốt cho tiếng Việt
- 384-dim vectors, nhỏ gọn (~120MB)
- Instruction-based: `"query: <q>"` và `"passage: <p>"`

**Offline fallback:** TF-IDF (char 2-4gram, 50k features) + TruncatedSVD (384 dims)
- Không cần internet, không cần GPU
- Chạy được trên máy hạn chế tài nguyên

### Hybrid Search với RRF

```
Dense Score (cosine)  ─┐
                        ├─→ Reciprocal Rank Fusion → Final Ranking
BM25 Score             ─┘
```

Công thức RRF: `score = Σ 1/(k + rank_i)` với `k=60`

BM25 giúp khớp từ khóa pháp lý chính xác (tên điều luật, nghị định, mức phạt).
Dense search giúp tìm ngữ nghĩa tương tự (synonym, paraphrase).

### Reranking Heuristics

| Factor                      | Điều chỉnh | Lý do                               |
|-----------------------------|:----------:|-------------------------------------|
| `chunk_type == "qa_pair"`   | ×1.15      | Câu hỏi khớp trực tiếp             |
| Keyword overlap với query   | +0.002×n   | Từ khóa quan trọng                  |
| `chunk_index == 0`          | ×1.05      | Chunk đầu có đầy đủ ngữ cảnh       |
| Có references               | ×1.03      | Đáng tin cậy hơn                   |
| Intent "penalty", no amount | ×0.92      | Penalty answer cần có số tiền cụ thể|

---

## Hướng cải thiện

1. **Neural Cross-Encoder Reranking** — thay heuristic bằng `cross-encoder/ms-marco-MiniLM-L-6-v2` để rerank chính xác hơn

2. **Query Expansion** — dùng LLM tạo 3-5 phiên bản query khác nhau (HyDE - Hypothetical Document Embedding)

3. **Hierarchical Chunking** — lưu cả chunk nhỏ (retrieval) và chunk to (context) để trade-off precision/recall

4. **Multi-vector Retrieval** — embed cả question và answer riêng biệt, search trên cả hai

5. **Incremental Indexing** — theo dõi `updated_at` của documents, chỉ re-embed docs mới/thay đổi

6. **Caching** — cache embedding của frequent queries để giảm latency

---

## Dataset

- **Nguồn:** Thư viện Pháp Luật Việt Nam (thuvienphapluat.vn)
- **Kích thước:** 6,820 rows (6,617 sau khi clean)
- **Cấu trúc:** `question`, `answer`, `references` (điều luật trích dẫn), `request_link`
- **Domain coverage:** Đường bộ, Ô tô, Xe máy, Bằng lái, Đăng ký xe, Đăng kiểm, Hàng không, Đường sắt, Đường thủy
