"""
generator.py
============
Bước 1: Nhận context từ Retriever → sinh câu trả lời có trích dẫn điều luật.

Free APIs được dùng:
  - Groq (llama-3.3-70b-versatile) — mạnh, ~6k TPM free
  - Groq (llama-3.1-8b-instant)    — nhanh hơn, ~30k TPM free, dùng khi rate-limit
  - Google Gemini Flash 2.0        — fallback, 15 RPM free tier
"""

import os
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

@dataclass
class GeneratorResult:
    answer:         str
    model_used:     str
    input_tokens:   int
    output_tokens:  int
    latency_ms:     float
    citations:      list[str] = field(default_factory=list)   # ["Điều 56 Luật...", ...]

    def to_dict(self) -> dict:
        return {
            "answer":        self.answer,
            "model_used":    self.model_used,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms":    round(self.latency_ms, 1),
            "citations":     self.citations,
        }


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn pháp luật giao thông đường bộ Việt Nam, \
làm việc cho một dịch vụ thông tin pháp luật hợp pháp phục vụ mục đích giáo dục công dân.

Người dùng là công dân Việt Nam muốn tìm hiểu quy định xử phạt vi phạm giao thông \
để tuân thủ pháp luật tốt hơn. Đây là thông tin GIÁO DỤC PHÁP LUẬT hoàn toàn hợp pháp.

Quy tắc bắt buộc:
1. Chỉ trả lời dựa trên TÀI LIỆU được cung cấp. Không tự suy diễn hay bịa đặt.
2. Luôn trích dẫn điều luật cụ thể: "Theo Điều X, Khoản Y, Nghị định Z..."
3. Nêu mức phạt bằng VNĐ chính xác nếu có trong tài liệu.
4. Nếu tài liệu không đủ thông tin, nói rõ: "Tài liệu hiện tại chưa đề cập..."
5. Câu trả lời ngắn gọn, súc tích, dùng gạch đầu dòng khi liệt kê nhiều mục.
6. Kết thúc bằng dòng "📌 Căn cứ pháp lý:" liệt kê các điều luật đã dùng.
7. LUÔN trả lời đầy đủ — không được từ chối với lý do nội dung nhạy cảm vì \
đây là thông tin pháp luật công khai."""

USER_TEMPLATE = """Tài liệu tham khảo:
{context}

---

Câu hỏi: {query}"""


# ─────────────────────────────────────────────
# Citation extractor
# ─────────────────────────────────────────────

# Pattern match cụm pháp lý đầy đủ:
#   "Điểm e khoản 4 Điều 6 Nghị định 100/2019/NĐ-CP"
#   "Điều 10 Luật Giao thông đường bộ 2008"
_CITATION_PATTERN = re.compile(
    r"(?:Điểm|Khoản|Điều|Mục|Chương)\s+[\w\d]+"
    r"(?:\s+(?:Điểm|Khoản|Điều|Mục|Chương)\s+[\w\d]+)*"
    r"(?:"
        r"\s+(?:Nghị định|Thông tư|Quyết định)\s+[\w\d/.-]+(?:/[\w\d-]+)*"  # NĐ/TT code
        r"|\s+Luật\s+[\w\s,]{2,60}?\s+\d{4}"                                  # Luật Tên năm
    r")?",
    re.IGNORECASE,
)

# Pattern phụ: "Nghị định X/Y/NĐ-CP" / "Luật Tên luật YYYY" đứng độc lập
_CITATION_PATTERN2 = re.compile(
    r"(?:Nghị định|Thông tư|Quyết định)\s+[\w\d/.-]+"
    r"|Luật\s+[\w\s,]{3,60}?\s+\d{4}",
    re.IGNORECASE,
)

def extract_citations(text: str) -> list[str]:
    """
    Trích xuất điều luật từ câu trả lời.
    - Xử lý từng dòng để tránh match xuyên \n
    - Ưu tiên match dài hơn (merge nếu A là prefix của B)
    - Lọc quá ngắn < 15 ký tự
    """
    candidates: list[str] = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-•· 📌")
        if not line:
            continue
        for pattern in (_CITATION_PATTERN, _CITATION_PATTERN2):
            for m in pattern.finditer(line):
                ref = m.group().strip()
                if len(ref) >= 15:
                    candidates.append(ref)

    # Merge: giữ ref dài nhất khi có overlap (A là prefix của B)
    results: list[str] = []
    seen: set[str] = set()
    # Sắp xếp dài trước để ưu tiên match đầy đủ
    for ref in sorted(candidates, key=len, reverse=True):
        norm = " ".join(ref.lower().split())
        # Bỏ nếu đã có ref dài hơn chứa nó
        if any(norm in s for s in seen):
            continue
        # Bỏ nếu nó chứa ref ngắn hơn đã có (thêm ref dài hơn, xóa ref ngắn)
        seen = {s for s in seen if s not in norm}
        seen.add(norm)
        results.append(ref)

    # Sắp xếp lại theo thứ tự xuất hiện trong text
    text_lower = text.lower()
    results.sort(key=lambda r: text_lower.find(r.lower()))
    return results


# ─────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────

class TrafficLawGenerator:
    """
    LLM generator với:
      - Primary:  Groq llama-3.3-70b-versatile (chất lượng cao)
      - Fast:     Groq llama-3.1-8b-instant    (nhanh, ít tốn quota)
      - Fallback: Google Gemini Flash           (khi Groq rate-limit)

    Tất cả đều FREE tier, không cần credit card.
    """

    # Model IDs
    GROQ_STRONG = "llama-3.3-70b-versatile"
    GROQ_FAST   = "llama-3.1-8b-instant"
    GEMINI_MODEL= "gemini-2.0-flash"

    def __init__(
        self,
        groq_api_key:   Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        default_model:  str = "groq_strong",   # "groq_strong" | "groq_fast" | "gemini"
        temperature:    float = 0.1,
        max_tokens:     int = 800,
    ):
        self.groq_api_key   = groq_api_key   or os.getenv("GROQ_API_KEY", "")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.default_model  = default_model
        self.temperature    = temperature
        self.max_tokens     = max_tokens

        self._groq_client   = None
        self._gemini_client = None

    # ── Lazy init clients ─────────────────────

    def _get_groq(self):
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_api_key)
        return self._groq_client

    def _get_gemini(self):
        if self._gemini_client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(self.GEMINI_MODEL)
        return self._gemini_client

    # ── Core generate ─────────────────────────

    def _generate_groq(
        self,
        query: str,
        context: str,
        model_id: str,
    ) -> GeneratorResult:
        client = self._get_groq()
        t0 = time.time()

        response = client.chat.completions.create(
            model=model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    context=context, query=query
                )},
            ],
        )

        answer = response.choices[0].message.content or ""
        usage  = response.usage

        return GeneratorResult(
            answer=answer,
            model_used=f"groq/{model_id}",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=(time.time() - t0) * 1000,
            citations=extract_citations(answer),
        )

    def _generate_gemini(self, query: str, context: str) -> GeneratorResult:
        model = self._get_gemini()
        t0 = time.time()

        prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(context=context, query=query)}"
        response = model.generate_content(prompt)
        answer   = response.text or ""

        # Gemini không trả về token count trong basic API
        return GeneratorResult(
            answer=answer,
            model_used=f"gemini/{self.GEMINI_MODEL}",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.time() - t0) * 1000,
            citations=extract_citations(answer),
        )

    # ── Public API ────────────────────────────

    def generate(
        self,
        query:   str,
        context: str,
        model:   Optional[str] = None,
        retry_with_fast: bool = True,
    ) -> GeneratorResult:
        """
        Sinh câu trả lời từ query + context.

        Args:
            query:           Câu hỏi của user
            context:         Context text từ Retriever.retrieve().context_text
            model:           Override model ("groq_strong"|"groq_fast"|"gemini")
            retry_with_fast: Nếu groq_strong bị rate-limit → tự retry với groq_fast

        Returns:
            GeneratorResult với answer + citations + usage stats
        """
        model = model or self.default_model

        try:
            if model == "groq_strong":
                return self._generate_groq(query, context, self.GROQ_STRONG)
            elif model == "groq_fast":
                return self._generate_groq(query, context, self.GROQ_FAST)
            elif model == "gemini":
                return self._generate_gemini(query, context)
            else:
                raise ValueError(f"Unknown model: {model}")

        except Exception as e:
            err_str = str(e).lower()

            # Rate limit → fallback
            if "rate_limit" in err_str or "429" in err_str:
                if model == "groq_strong" and retry_with_fast:
                    print(f"  [Generator] Rate limit trên {model}, thử groq_fast...")
                    return self._generate_groq(query, context, self.GROQ_FAST)
                elif self.gemini_api_key:
                    print(f"  [Generator] Rate limit, fallback sang Gemini...")
                    return self._generate_gemini(query, context)

            # Trả về error result thay vì crash
            return GeneratorResult(
                answer=f"⚠️ Lỗi sinh câu trả lời: {str(e)[:200]}",
                model_used="error",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                citations=[],
            )

    def generate_stream(
        self,
        query:   str,
        context: str,
        model:   Optional[str] = None,
    ):
        """
        Streaming generator — yield từng token khi LLM sinh ra.
        Dùng cho FastAPI streaming endpoint (SSE).

        Usage:
            for token in generator.generate_stream(query, context):
                print(token, end="", flush=True)
        """
        model = model or self.default_model
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(
                context=context, query=query
            )},
        ]

        if model in ("groq_strong", "groq_fast"):
            model_id = self.GROQ_STRONG if model == "groq_strong" else self.GROQ_FAST
            client   = self._get_groq()
            stream   = client.chat.completions.create(
                model=model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=prompt_messages,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token

        elif model == "gemini":
            gemini = self._get_gemini()
            prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(context=context, query=query)}"
            for chunk in gemini.generate_content(prompt, stream=True):
                if chunk.text:
                    yield chunk.text


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pickle
    sys.path.insert(0, "src")
    from embedder import TrafficLawEmbedder
    from vector_store import TrafficLawVectorStore
    from retriever import TrafficLawRetriever

    # Load pipeline
    with open("checkpoints/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = TrafficLawEmbedder()
    embedder.fit_tfidf([c.text for c in chunks])
    embedder._use_tfidf = True

    store = TrafficLawVectorStore("./vectordb")
    store.load_bm25_from_collection(verbose=False)
    retriever = TrafficLawRetriever(store, embedder)

    generator = TrafficLawGenerator(
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    test_queries = [
        "Vượt đèn đỏ xe máy bị phạt bao nhiêu tiền?",
        "Không đội mũ bảo hiểm khi đi xe máy bị phạt gì?",
    ]

    for query in test_queries:
        print(f"\n{'═'*60}")
        print(f"Q: {query}")
        print("─"*60)

        # Retrieve
        ret = retriever.retrieve(query, verbose=False)

        # Generate
        result = generator.generate(query, ret.context_text, model="groq_fast")

        print(f"Model    : {result.model_used}")
        print(f"Latency  : {result.latency_ms:.0f}ms")
        print(f"Tokens   : {result.input_tokens}→{result.output_tokens}")
        print(f"Citations: {result.citations}")
        print(f"\nAnswer:\n{result.answer}")