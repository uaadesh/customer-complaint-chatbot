# chatbot/rag_retriever.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from typing import Optional

def build_retriever(
    doc_path: str = "knowledge-base/faq.txt",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    Builds a retriever for RAG by:
    - Loading the knowledge base document
    - Splitting it into chunks
    - Embedding chunks using a Hugging Face model
    - Indexing using FAISS for similarity search

    Returns:
        retriever (BaseRetriever): FAISS-based document retriever
    """
    try:
        # Load the document(s)
        loader = TextLoader(doc_path)
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents from {doc_path}")
    except Exception as e:
        logging.error(f"Failed to load documents: {e}")
        raise

    try:
        # Split documents into overlapping chunks
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)

        # Encode chunks using Hugging Face sentence transformer
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Store embeddings in FAISS for similarity search
        vectordb = FAISS.from_documents(chunks, embeddings)
        retriever = vectordb.as_retriever()
        return retriever
    except Exception as e:
        logging.error(f"Failed to build retriever: {e}")
        raise
