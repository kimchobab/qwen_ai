import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_chunks_from_pdf(path=settings.PDF_PATH):
    """
    PDF 파일을 로드하고, 텍스트를 chunk 단위로 나눠서 반환
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.OVERLAP
    )
    return splitter.split_documents(docs)
