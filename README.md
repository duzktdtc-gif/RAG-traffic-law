# RAG Luật Giao thông Việt Nam

Hệ thống hỏi-đáp pháp luật giao thông đường bộ sử dụng **Retrieval-Augmented Generation (RAG)**, xây dựng trên dataset 6.000 cặp QA từ các văn bản pháp luật Việt Nam.

> **Demo**: Nhập câu hỏi bằng tiếng Việt tự nhiên → hệ thống truy xuất điều luật liên quan → LLM sinh câu trả lời có trích dẫn cụ thể.

---

##  Kết quả Evaluation

| Mode | Hit@5 | MRR@5 | NDCG@5 | Latency |
|------|-------|-------|--------|---------|
| **Hybrid** ✅ | **100%** | **0.983** | **0.988** | 226ms |
| BM25 | 100% | 0.937 | 0.953 | 197ms |
| Dense (TF-IDF) | 0% | 0.000 | 0.000 | 79ms |

Evaluated trên 90 samples, stratified across 15 domain tags.

> Dense search đạt 0% do dùng TF-IDF fallback thay vì neural embedding (`multilingual-e5-small`). Khi thay bằng neural model, dense search sẽ cạnh tranh được với hybrid.

---

##  Kiến trúc hệ thống

```
CSV (6,000 QA)
    │
    ▼
┌─────────────┐     3 chiến lược chunking:
│  DataLoader │ ──► QA-Pair / Sliding-Window / Article-Split
└─────────────┘
    │
    ▼
┌─────────────┐     TF-IDF + SVD 384d (fallback)
│   Embedder  │ ──► intfloat/multilingual-e5-small (neural)
└─────────────┘
    │
    ▼
┌──────────────┐     ChromaDB (dense) + BM25 (sparse)
│  VectorStore │ ──► Hybrid search với RRF fusion
└──────────────┘
    │
    ▼
┌──────────────┐     Query expansion + Intent detection
│   Retriever  │ ──► Dedup + Cross-Encoder rerank
└──────────────┘
    │
    ▼
┌──────────────────┐     HyDE + ConversationMemory
│  AdvancedRAG     │ ──► Multi-turn coreference resolution
└──────────────────┘
    │
    ▼
┌──────────────┐     Groq Llama 3.1 8B / 3.3 70B
│  Generator   │ ──► Gemini Flash fallback
└──────────────┘
    │
    ▼
  Streamlit UI / FastAPI
```

---

## Tính năng nổi bật

### Chunking thông minh (3 chiến lược)
- **QA-Pair**: answer ngắn ≤ 800 ký tự → ghép question + answer thành 1 chunk
- **Article-Split**: answer có cấu trúc Điều/Khoản → tách theo ranh giới điều luật
- **Sliding-Window**: answer dài, không có cấu trúc → cửa sổ chồng lấp 64 tokens

### Hybrid Search
- **Dense**: embedding vector (cosine similarity)
- **BM25**: char n-gram (2-4) + bigram tokenization, pre-computed TF
- **Fusion**: Reciprocal Rank Fusion với alpha tunable

### Advanced RAG Pipeline
- **HyDE** (Hypothetical Document Embedding): LLM tạo câu trả lời giả → embed → search
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank top-k
- **Conversation Memory**: rule-based coreference resolution, inject lịch sử vào query

### Multi-turn conversation
```
User: "Vượt đèn đỏ xe máy bị phạt bao nhiêu?"
Bot:  "Phạt 800.000 – 1.000.000đ, tước GPLX 1-3 tháng..."

User: "Thế còn ô tô thì sao?"
      ↓ rewrite: "[Ngữ cảnh: Vượt đèn đỏ xe máy...] Thế còn ô tô thì sao?"
Bot:  "Ô tô vượt đèn đỏ bị phạt 4.000.000 – 6.000.000đ..."

```
![Khung xác nhận kế hoạch](img\a.png)
---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vector DB | ChromaDB |
| Embedding | TF-IDF+SVD 384d / `multilingual-e5-small` |
| Sparse Search | BM25 (custom, thuần Python) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq (Llama 3.1 8B, Llama 3.3 70B) |
| LLM Fallback | Google Gemini 2.0 Flash |
| UI | Streamlit |
| Dataset | 6,000 QA pairs — luật giao thông VN |

---

## Cài đặt và chạy

### 1. Clone và cài thư viện

```bash
git clone https://github.com/<your-username>/rag_traffic.git
cd rag_traffic
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Cấu hình API key

```bash
cp .env.example .env
```

Mở `.env` và điền:
```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here   # optional
```

Lấy Groq API key miễn phí tại: https://console.groq.com

### 3. Build index (chạy một lần)

```bash
python src/indexing_pipeline.py --csv dataset/Dataset_6000.csv
```

Quá trình: Load CSV → Chunk → Embed → Index vào ChromaDB + BM25.
Thời gian: ~5-10 phút. Kết quả lưu vào `checkpoints/` và `vectordb/`.

### 4. Chạy giao diện chat

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

### 5. Chạy evaluation (tùy chọn)

```bash
python src/evaluator.py
```

---

## Cấu trúc project

```
rag_traffic/
├── src/
│   ├── data_loader.py        # Load & validate CSV → Document
│   ├── chunker.py            # 3-strategy chunking
│   ├── embedder.py           # TF-IDF / neural embedding
│   ├── vector_store.py       # ChromaDB + BM25 index
│   ├── retriever.py          # Hybrid search + rerank
│   ├── generator.py          # LLM generation + citation extraction
│   ├── advanced_rag.py       # HyDE + CrossEncoder + Memory
│   ├── evaluator.py          # Hit Rate / MRR / NDCG evaluation
│   └── indexing_pipeline.py  # End-to-end build pipeline
├── dataset/
│   └── Dataset_6000.csv
├── checkpoints/              # chunks.pkl, embeddings.npz (auto-generated)
├── vectordb/                 # ChromaDB files (auto-generated)
├── app.py                    # Streamlit UI
├── .env.example
├── requirements.txt
└── README.md
```

---

## Hướng phát triển

- [ ] Thay TF-IDF bằng `intfloat/multilingual-e5-small` để cải thiện dense search
- [ ] Bật HyDE cho câu hỏi ngắn, mơ hồ
- [ ] Thêm paraphrase queries vào eval set để đo generalization
- [ ] FastAPI endpoint để tích hợp với frontend khác
- [ ] Docker container cho deployment

---
