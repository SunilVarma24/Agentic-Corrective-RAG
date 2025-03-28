# src/document.py

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# Load & Chunk the Document
def load_and_chunk_document(file_path: str, chunk_size: int = 2000, chunk_overlap: int = 300):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(docs)
    print(len(chunked_docs))
    return chunked_docs

# Define the Chroma DB & Retriever
def create_chroma_retriever(chunked_docs, embed_model, persist_dir: str = "./doc_db", collection_name: str = "rag_db", score_threshold: float = 0.3, k: int = 3):
    chroma_db = Chroma.from_documents(
        documents=chunked_docs,
        collection_name=collection_name,
        embedding=embed_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=persist_dir
    )
    
    similarity_threshold_retriever = chroma_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    return similarity_threshold_retriever
