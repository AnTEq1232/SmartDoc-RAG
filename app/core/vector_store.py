import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

def add_documents_to_store(splits):
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PATH
    )
    return vectorstore