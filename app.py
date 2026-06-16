"""
app.py
======
Giao diện chat Streamlit cho hệ thống RAG Luật Giao thông VN.

Chạy:
    streamlit run app.py
"""

import os
import sys
import time
import pickle
import streamlit as st

sys.path.insert(0, "src")

# ─────────────────────────────────────────────
# Page config — phải gọi trước mọi st.*
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Tư vấn Luật Giao thông VN",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Be+Vietnam+Pro:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', 'Inter', sans-serif;
}

/* Hide default header */
#MainMenu, header, footer { visibility: hidden; }

/* Main background */
.main { background: #F7F8FA; }

/* Chat message bubbles */
.user-bubble {
    background: #1B3A6B;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 15px;
    line-height: 1.5;
}
.bot-bubble {
    background: white;
    color: #1A1A2E;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 88%;
    font-size: 15px;
    line-height: 1.7;
    border: 1px solid #E8EAF0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Citation badges */
.citation-badge {
    display: inline-block;
    background: #EEF2FF;
    color: #3730A3;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 2px 3px;
    border: 1px solid #C7D2FE;
}

/* Stats bar */
.stats-bar {
    font-size: 11px;
    color: #9CA3AF;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #F0F0F0;
}

/* Source card */
.source-card {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 12px;
    color: #6B7280;
}
.source-score {
    display: inline-block;
    background: #DCFCE7;
    color: #15803D;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 11px;
    margin-right: 6px;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #1B3A6B 0%, #2563EB 100%);
    color: white;
    padding: 20px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.app-header h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.app-header p {
    margin: 4px 0 0;
    font-size: 13px;
    opacity: 0.8;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #1B3A6B;
}
.metric-label {
    font-size: 11px;
    color: #9CA3AF;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Input styling */
.stTextInput input {
    border-radius: 24px !important;
    border: 2px solid #E5E7EB !important;
    padding: 10px 18px !important;
    font-size: 15px !important;
}
.stTextInput input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* Thinking spinner */
.thinking {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #6B7280;
    font-size: 14px;
    padding: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load pipeline (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    from embedder import TrafficLawEmbedder
    from vector_store import TrafficLawVectorStore
    from retriever import TrafficLawRetriever
    from generator import TrafficLawGenerator
    from advanced_rag import AdvancedRAGPipeline

    with open("checkpoints/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = TrafficLawEmbedder()
    embedder.fit_tfidf([c.text for c in chunks])
    embedder._use_tfidf = True

    store = TrafficLawVectorStore("./vectordb")
    store.load_bm25_from_collection(verbose=False)

    retriever = TrafficLawRetriever(store, embedder)
    generator = TrafficLawGenerator(groq_api_key=os.getenv("GROQ_API_KEY"))

    pipeline = AdvancedRAGPipeline(
        retriever=retriever,
        generator=generator,
        embedder=embedder,
        use_hyde=False,
        use_reranker=True,
        use_memory=True,
    )
    return pipeline


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "avg_latency" not in st.session_state:
    st.session_state.avg_latency = []


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚖️ Luật Giao thông VN")
    st.markdown("Hệ thống RAG tư vấn pháp luật đường bộ")
    st.divider()

    # Model selector
    model_choice = st.selectbox(
        "🤖 Model",
        ["groq_fast", "groq_strong"],
        format_func=lambda x: {
            "groq_fast":   "Llama 3.1 8B (nhanh)",
            "groq_strong": "Llama 3.3 70B (mạnh)",
        }[x],
    )

    # Search mode
    search_mode = st.selectbox(
        "🔍 Chế độ tìm kiếm",
        ["hybrid", "bm25", "dense"],
        format_func=lambda x: {
            "hybrid": "Hybrid (BM25 + Dense)",
            "bm25":   "BM25 (từ khóa)",
            "dense":  "Dense (ngữ nghĩa)",
        }[x],
    )

    st.divider()

    # Session stats
    st.markdown("**📊 Phiên hiện tại**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Câu hỏi", st.session_state.total_queries)
    with col2:
        avg_lat = (
            sum(st.session_state.avg_latency) / len(st.session_state.avg_latency)
            if st.session_state.avg_latency else 0
        )
        st.metric("Avg latency", f"{avg_lat:.0f}ms")

    st.divider()

    # Evaluation results
    st.markdown("**🏆 Kết quả Evaluation**")
    st.markdown("""
    | Metric | Hybrid |
    |--------|--------|
    | Hit@5 | **100%** |
    | MRR@5 | **0.983** |
    | NDCG@5 | **0.988** |
    | Latency | **226ms** |
    """)

    st.divider()

    # Example questions
    st.markdown("**💡 Câu hỏi mẫu**")
    examples = [
        "Vượt đèn đỏ xe máy bị phạt bao nhiêu?",
        "Uống rượu lái xe bị phạt gì?",
        "Không đội mũ bảo hiểm bị phạt bao nhiêu tiền?",
        "Xe máy không có gương chiếu hậu bị phạt không?",
        "Đi ngược chiều bị phạt bao nhiêu?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.pending_query = ex

    st.divider()

    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.avg_latency = []
        st.rerun()

    # Tech stack
    st.markdown("**🔧 Tech Stack**")
    st.caption("""
    - **Vector DB**: ChromaDB
    - **Embedding**: TF-IDF + SVD (384d)
    - **Search**: BM25 + Dense Hybrid
    - **Reranker**: Cross-Encoder MiniLM
    - **Memory**: Conversation context
    - **LLM**: Groq (Llama 3.1/3.3)
    - **Dataset**: 6,000 QA pairs
    """)


# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <h1>⚖️ Tư vấn Luật Giao thông Việt Nam</h1>
    <p>Hỏi bất kỳ câu hỏi nào về luật giao thông đường bộ — hệ thống sẽ tra cứu và trích dẫn điều luật chính xác</p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
with st.spinner("Đang tải hệ thống..."):
    try:
        pipeline = load_pipeline()
        pipeline_loaded = True
    except Exception as e:
        st.error(f"Lỗi tải pipeline: {e}")
        pipeline_loaded = False

# Chat history
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color: #9CA3AF;">
            <div style="font-size: 48px; margin-bottom: 12px;">⚖️</div>
            <div style="font-size: 16px; font-weight: 600; color: #374151; margin-bottom: 8px;">
                Xin chào! Tôi có thể giúp bạn tra cứu luật giao thông.
            </div>
            <div style="font-size: 14px;">
                Hỏi về mức phạt, quy định, thủ tục... — tôi sẽ trích dẫn điều luật cụ thể.
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Bot message
            answer_html = msg["content"].replace("\n", "<br>")
            citations_html = ""
            if msg.get("citations"):
                citations_html = "<div style='margin-top:10px'>"
                for c in msg["citations"]:
                    citations_html += f'<span class="citation-badge">📌 {c}</span>'
                citations_html += "</div>"

            stats = msg.get("stats", {})
            stats_html = ""
            if stats:
                stats_html = f"""
                <div class="stats-bar">
                    ⏱ {stats.get('total_ms', 0):.0f}ms &nbsp;·&nbsp;
                    🔍 {stats.get('retrieval_ms', 0):.0f}ms retrieval &nbsp;·&nbsp;
                    ✍️ {stats.get('generate_ms', 0):.0f}ms generate &nbsp;·&nbsp;
                    🪙 {stats.get('input_tokens', 0)}→{stats.get('output_tokens', 0)} tokens
                </div>
                """

            st.markdown(
                f'<div class="bot-bubble">{answer_html}{citations_html}{stats_html}</div>',
                unsafe_allow_html=True,
            )

            # Sources expander
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} nguồn tham khảo", expanded=False):
                    for src in msg["sources"]:
                        score_pct = int(src["score"] * 100) if src["score"] <= 1 else min(int(src["score"] * 10), 100)
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-score">{src["score"]:.3f}</span>'
                            f'<strong>{src["question"][:80]}...</strong><br>'
                            f'<span style="color:#9CA3AF">Type: {src["chunk_type"]}</span>'
                            f'{"&nbsp;·&nbsp;<a href=" + src["source_url"] + " target=_blank>Link</a>" if src.get("source_url") and src["source_url"] != "nan" else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

# Xử lý example button click
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": pending})
    with st.spinner("Đang tra cứu..."):
        result = pipeline.ask(pending, search_mode=search_mode, model=model_choice)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
        "stats": result["stats"],
    })
    st.session_state.total_queries += 1
    st.session_state.avg_latency.append(result["stats"]["total_ms"])
    st.rerun()

# Text input
with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Câu hỏi",
            placeholder="Ví dụ: Vượt đèn đỏ xe máy bị phạt bao nhiêu tiền?",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Gửi ➤", use_container_width=True)

if submitted and user_input.strip() and pipeline_loaded:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Đang tra cứu điều luật..."):
        t0 = time.time()
        result = pipeline.ask(
            user_input,
            search_mode=search_mode,
            model=model_choice,
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
        "stats": result["stats"],
    })
    st.session_state.total_queries += 1
    st.session_state.avg_latency.append(result["stats"]["total_ms"])
    st.rerun()