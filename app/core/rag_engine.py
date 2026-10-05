import os
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.genai.errors import APIError, ClientError


def get_conversational_rag_chain(retriever, llm):
    """Tworzy dwuetapowy łańcuch RAG uwzględniający historię konwersacji."""
    # 1. Prompt odpowiedzialny za przekształcanie pytań z kontekstem na samodzielne zdania
    contextualize_q_system_prompt = (
        "Biorąc pod uwagę historię rozmowy oraz najnowsze pytanie użytkownika, "
        "które może odwoływać się do kontekstu w historii, sformułuj samodzielne pytanie, "
        "które można zrozumieć bez znajomości historii rozmowy. NIE odpowiadaj na pytanie, "
        "jedynie sformułuj je ponownie w razie potrzeby, w przeciwnym razie zwróć je bez zmian."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Retriever reformułujący zapytanie użytkownika z użyciem historii
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Prompt odpowiedzialny za właściwą odpowiedź na podstawie pobranego kontekstu
    system_prompt = (
        "Jesteś asystentem odpowiadającym na pytania na podstawie dołączonych dokumentów.\n"
        "Wykorzystaj poniższe fragmenty kontekstu, aby odpowiedzieć na pytanie.\n"
        "Jeśli nie znasz odpowiedzi na podstawie kontekstu, powiedz szczerze, że nie wiesz.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Główny łańcuch łączący retriever ze świadomością historii oraz odpowiedź na dokumentach
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_llm():
    """Inicjalizuje lekki i stabilny model Gemini wraz z obsługą błędów i ponowień."""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_retries=6,  # Automatyczne ponawianie zapytań przy chwilowym błędzie 503
        )
    except Exception as e:
        raise RuntimeError(f"Błąd podczas inicjalizacji modelu Gemini LLM: {str(e)}") from e


def get_vectorstore(persist_directory="./chroma_db"):
    """Pobiera istniejącą bazę ChromaDB z osadzeniami Google Generative AI."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    except (ClientError, APIError) as e:
        raise RuntimeError(f"Błąd API Google podczas ładowania baz danych wektorowych: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Nie udało się załadować bazy ChromaDB ({persist_directory}): {str(e)}") from e