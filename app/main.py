import streamlit as st
import os
from core.rag_engine import process_document, get_rag_chain
from config import DATA_DIR

st.set_page_config(page_title="SmartDoc RAG", page_icon="📄", layout="wide")

st.title("📄 SmartDoc RAG")

with st.sidebar:
    st.header("1. Wgraj dokument")
    uploaded_file = st.file_uploader("Wybierz plik PDF lub TXT", type=["pdf", "txt"])

    if uploaded_file is not None:
        os.makedirs(DATA_DIR, exist_ok=True)
        temp_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Przetwórz dokument"):
            with st.spinner("Przetwarzanie dokumentu..."):
                try:
                    process_document(temp_path)
                    st.success("Dokument został pomyślnie przetworzony!")
                except Exception as e:
                    st.error(f"Błąd: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Zadaj pytanie dotyczące wgranego dokumentu..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    rag_chain = get_rag_chain()
    
    if rag_chain is None:
        with st.chat_message("assistant"):
            res = "Najpierw wgraj i przetwórz dokument w panelu bocznym."
            st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Szukam odpowiedzi..."):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)

                with st.expander("Zobacz źródła z dokumentu"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.write(f"**Fragment {i+1}:**")
                        st.caption(doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": answer})