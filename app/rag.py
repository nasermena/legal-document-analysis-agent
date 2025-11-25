import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = "data/chroma"

def _get_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def ingest_text(raw_text: str) -> str:
    doc_id = str(uuid.uuid4())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(raw_text)

    docs = [Document(page_content=c, metadata={"doc_id": doc_id}) for c in chunks]
    db = _get_db()
    db.add_documents(docs)
    db.persist()
    return doc_id

def retrieve(doc_id: str, question: str, k: int = 4) -> List[Document]:
    db = _get_db()
    return db.similarity_search(question, k=k, filter={"doc_id": doc_id})
