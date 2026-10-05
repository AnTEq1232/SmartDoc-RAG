import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from app.core.rag_engine import (
    get_conversational_rag_chain,
    get_llm,
)


def _has_text(value: str) -> bool:
    return bool(value and value.strip())

# Konfiguracja strony
st.set_page_config(page_title="SmartDoc-RAG", page_icon="📄", layout="wide")
st.title("📄 SmartDoc-RAG — Czat z Twoim Dokumentem")

# --- 1. Inicjalizacja stanu sesji Streamlit ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- 2. Sidebar: Wgrywanie dokumentu PDF ---
with st.sidebar:
    st.header("⚙️ Zarządzanie dokumentem")
    uploaded_file = st.file_uploader("Prześlij plik PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Przetwórz i załaduj PDF"):
            with st.spinner("Przetwarzanie pliku PDF i tworzenie bazy wektorowej..."):
                # Zapis pliku tymczasowo na dysku
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Ładowanie i podział dokumentu na fragmenty
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()

                # Odfiltruj puste strony, aby uniknąć pustych embeddingów w Chroma.
                docs = [doc for doc in docs if _has_text(doc.page_content)]
                if not docs:
                    os.remove(tmp_file_path)
                    st.error("Nie udało się odczytać tekstu z PDF (puste strony lub skan bez OCR).")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)

                # Odfiltruj puste fragmenty po podziale.
                splits = [chunk for chunk in splits if _has_text(chunk.page_content)]
                if not splits:
                    os.remove(tmp_file_path)
                    st.error("Po podziale dokumentu nie znaleziono fragmentów z tekstem do indeksowania.")
                    st.stop()

                # Tworzenie bazy wektorowej w pamięci sesji
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings
                )

                st.session_state.vectorstore = vectorstore
                # Czyszczenie starej historii przy wgraniu nowego pliku
                st.session_state.messages = []
                os.remove(tmp_file_path)

                st.success("Plik przetworzony pomyślnie! Możesz zadać pytanie.")

    if st.session_state.vectorstore is not None:
        if st.button("🗑️ Wyczyść historię i plik"):
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.rerun()

# --- 3. Wyświetlanie historii czatu ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Pomocnicza konwersja historii wiadomości dla LangChain
def get_langchain_chat_history():
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history

# --- 4. Obsługa czatu ---
if st.session_state.vectorstore is None:
    st.info("👈 Prześlij plik PDF w panelu bocznym po lewej stronie, aby rozpocząć rozmowę.")
else:
    user_input = st.chat_input("Zadaj pytanie dotyczące wgranego dokumentu...")

    if user_input:
        # Wyświetlenie zapytania w UI i zapis do historii
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Przygotowanie łańcucha RAG
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = get_llm()
        rag_chain = get_conversational_rag_chain(retriever, llm)

        chat_history = get_langchain_chat_history()

        # Odpowiedź asystenta
        with st.chat_message("assistant"):
            with st.spinner("Szukam odpowiedzi w dokumencie..."):
                try:
                    response = rag_chain.invoke({
                        "input": user_input,
                        "chat_history": chat_history,
                    })
                    answer = response["answer"]
                    st.markdown(answer)

                    # Zapis odpowiedzi
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Wystąpił błąd: {e}")