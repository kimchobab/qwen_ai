import settings
print(" settings 모듈 경로:", settings.__file__)
print(" PDF_PATH 값:", getattr(settings, "PDF_PATH", " 없음"))

from pdf_utils import extract_chunks_from_pdf
from vector_store import create_vector_db, search_db
from qwen_infer import ask_qwen

def main():
    # 1. PDF에서 텍스트 chunk 추출
    print("📄 PDF에서 텍스트 추출 중...")
    chunks = extract_chunks_from_pdf()

    # 2. 벡터 DB 생성
    print(" 벡터 DB 생성 중...")
    db = create_vector_db(chunks)

    # 3. 사용자 입력 받기
    print(" 질문을 입력하세요. (종료하려면 'exit')")
    while True:
        query = input(" 질문: ")
        if query.lower() in ["exit", "quit"]:
            break

        # 4. 관련 문서 검색
        relevant_chunks = search_db(db, query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_chunks])

        # 5. Qwen에게 질문하기
        print(" Qwen이 답변 중...\n")
        answer = ask_qwen(context, query)
        print(" 답변:", answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
