from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import settings

def create_vector_db(chunks):
    """
    텍스트 조각(chunk)을 임베딩 후 FAISS 벡터 DB 생성
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    return db

def search_db(db, query, k=3):
    """
    쿼리와 가장 유사한 문서 k개 검색
    """
    return db.similarity_search(query, k=k)
