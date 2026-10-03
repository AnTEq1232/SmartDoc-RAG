import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from core.vector_store import get_vector_store, add_documents_to_store, CHROMA_PATH

def process_document(file_path: str):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Obsługiwane są tylko pliki PDF i TXT.")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    return add_documents_to_store(splits)

def get_rag_chain():
    """Tworzy i zwraca łańcuch RAG."""
    if not os.path.exists(CHROMA_PATH):
        return None

    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = (
        "Jesteś pomocnym asystentem do analizy dokumentów.\n"
        "Odpowiadaj na pytania wyłącznie na podstawie poniższego kontekstu.\n"
        "Jeśli nie znasz odpowiedzi na podstawie kontekstu, powiedz szczerze: 'Nie znajduję tej informacji w dostarczonych dokumentach.'\n\n"
        "Kontekst:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)