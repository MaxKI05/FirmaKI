import os
import streamlit as st
import openai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── Prompt-Vorlagen für strukturierte Antworten mit Inline-Quellen
QUESTION_PROMPT_TEMPLATE = """
Du bist ein hilfsbereiter Assistent. Nutze ausschließlich den folgenden Textauszug als Quelle.

{context}

Frage:
{question}

Antwort:
Bitte strukturiere deine Antwort mit klaren Markdown-Überschriften (##) und füge nach jeder Aussage eine Quellenangabe in der Form (Seite X) ein.
"""

COMBINE_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Assistent. Du bekommst mehrere kurze Antworten (Summaries) zu einer Frage.
Fasse sie zu einer präzisen Antwort zusammen, strukturiere sie mit Markdown-Überschriften (##) und füge nach jeder Aussage (Seite X) als Quelle ein.
"""

@st.cache_resource
# Lade und cache die RetrievalQA-Chain

def load_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    index_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(index_path):
        st.error("Index-Ordner 'leitfaden_index' nicht gefunden.")
        st.stop()
    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "maximal_marginal_relevance": True
    })
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    question_prompt = PromptTemplate(
        template=QUESTION_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    combine_prompt = PromptTemplate(
        template=COMBINE_PROMPT_TEMPLATE,
        input_variables=["question", "summaries"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt
        },
        return_source_documents=True
    )

# Hauptfunktion

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")

    # API-Key prüfen
    openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        st.error("🔑 API-Schlüssel fehlt. Bitte in Streamlit Secrets hinterlegen.")
        st.stop()

    # Session-State initialisieren
    if "history" not in st.session_state:
        st.session_state.history = []  # List of dicts {question, answer}

    # Statisches Formular oben
    st.title("📘 Frag den Betreiberleitfaden")
    question = st.text_input("❓ Deine Frage:", key="input")
    if st.button("🔍 Antwort anzeigen"):
        if question.strip():
            chain = load_chain()
            with st.spinner("📚 Ich durchsuche den Leitfaden..."):
                result = chain({"query": question})
            answer = result.get("result")
            # Antwort speichern
            st.session_state.history.append({"question": question, "answer": answer})

    st.markdown("---")
    # Antworten darunter anzeigen
    for entry in st.session_state.history:
        st.markdown(f"**Du:** {entry['question']}")
        st.markdown(entry['answer'])
        st.markdown("---")

# Entry Point
if __name__ == "__main__":
    main()
