import os
import csv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# === 1. CẤU HÌNH GROQ ===
# Điền API Key của bạn vào đây
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0  # Bắt buộc = 0 để chấm điểm khách quan
)


# === 2. ĐỊNH NGHĨA CẤU TRÚC KẾT QUẢ (Structured Output) ===
# Ép Groq phải trả về đúng format JSON này, không được chém gió
class EvalResult(BaseModel):
    score: float = Field(description="Điểm đánh giá từ 0.0 đến 1.0")
    reason: str = Field(description="Lý do ngắn gọn bằng tiếng Việt, chỉ ra lỗi nếu có")


# Bind cấu trúc output vào LLM (Groq hỗ trợ việc này cực tốt)
evaluator_llm = llm.with_structured_output(EvalResult)

# === 3. THIẾT LẬP CÁC PROMPT ĐÁNH GIÁ ===
prompt_faithfulness = ChatPromptTemplate.from_template("""
Bạn là một giám khảo nghiêm ngặt đánh giá hệ thống RAG Luật Giao thông.
Nhiệm vụ: Kiểm tra xem [Câu trả lời] có hoàn toàn dựa trên [Ngữ cảnh] hay không.
Nếu [Câu trả lời] bịa đặt thông tin (ảo giác), tự suy diễn mức phạt không có trong [Ngữ cảnh], hãy trừ điểm nặng.

[Câu hỏi]: {query}
[Ngữ cảnh]: {context}
[Câu trả lời]: {answer}

Hãy chấm điểm độ trung thực (Faithfulness) từ 0.0 đến 1.0 và giải thích lý do.
""")

prompt_relevancy = ChatPromptTemplate.from_template("""
Bạn là một giám khảo nghiêm ngặt đánh giá hệ thống RAG.
Nhiệm vụ: Kiểm tra xem [Câu trả lời] có giải quyết trực tiếp và đúng trọng tâm [Câu hỏi] hay không.
Nếu câu trả lời lan man, copy-paste nguyên đoạn luật dài dòng thay vì tóm tắt, hoặc lạc đề, hãy trừ điểm.

[Câu hỏi]: {query}
[Câu trả lời]: {answer}

Hãy chấm điểm độ liên quan (Relevancy) từ 0.0 đến 1.0 và giải thích lý do.
""")


# === 4. HÀM ĐÁNH GIÁ TỪNG SAMPLE ===
def evaluate_sample(query: str, context: str, answer: str):
    # Đánh giá Faithfulness (Độ trung thực với Context)
    chain_faith = prompt_faithfulness | evaluator_llm
    res_faith = chain_faith.invoke({"query": query, "context": context, "answer": answer})

    # Đánh giá Relevancy (Độ liên quan với Query)
    chain_rel = prompt_relevancy | evaluator_llm
    res_rel = chain_rel.invoke({"query": query, "answer": answer})

    return {
        "faithfulness_score": res_faith.score,
        "faithfulness_reason": res_faith.reason,
        "relevancy_score": res_rel.score,
        "relevancy_reason": res_rel.reason
    }


# === 5. LUỒNG CHẠY CHÍNH ===
if __name__ == "__main__":
    # Dữ liệu giả lập (Bạn hãy thay bằng vòng lặp lấy dữ liệu thật từ Retriever + LLM của bạn)
    eval_data = [
        {
            "query": "Vượt đèn đỏ bị phạt bao nhiêu tiền?",
            "context": "Người đi xe máy vượt đèn đỏ bị phạt tiền từ 800.000 đồng đến 1.000.000 đồng.",
            "answer": "Theo quy định, người đi xe máy vượt đèn đỏ sẽ bị phạt từ 800.000đ đến 1.000.000đ."
        },
        {
            "query": "Không có bằng lái xe máy bị phạt thế nào?",
            "context": "Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với người điều khiển xe mô tô không có GPLX.",
            # Câu trả lời này bị "ảo giác" thêm chữ "bị giam xe" không có trong context -> Faithfulness sẽ thấp
            "answer": "Bạn sẽ bị phạt 1 đến 2 triệu đồng và có thể bị giam xe."
        }
    ]

    print("=== BẮT ĐẦU ĐÁNH GIÁ GENERATOR (CUSTOM GROQ EVALUATOR) ===\n")

    results = []
    for i, sample in enumerate(eval_data):
        print(f"Đang chấm mẫu {i + 1}/{len(eval_data)}: {sample['query'][:40]}...")

        scores = evaluate_sample(
            query=sample["query"],
            context=sample["context"],
            answer=sample["answer"]
        )

        # In kết quả trực quan ra màn hình
        print(f"  -> 🎯 Faithfulness: {scores['faithfulness_score']:.2f} | 🎯 Relevancy: {scores['relevancy_score']:.2f}")
        if scores['faithfulness_score'] < 1.0:
            print(f"     ⚠️ Cảnh báo ảo giác: {scores['faithfulness_reason']}")

        results.append({**sample, **scores})

    # Lưu kết quả chi tiết ra file CSV để mở bằng Excel
    csv_file = "eval_custom_results.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Hoàn thành! Đã lưu kết quả chi tiết vào '{csv_file}'")
