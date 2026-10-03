import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = "./chroma_db"

def get_vector_store():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def add_documents_to_store(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore