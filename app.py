import os
import sys
import streamlit as st

# ─── Sicherstellen, dass tiktoken installiert ist (sonst freundliche Fehlermeldung) ───
try:
    import tiktoken
except ImportError:
    st.set_page_config(page_title="Fehler", layout="centered")
    st.title("📦 Fehlendes Paket: tiktoken")
    st.error("Bitte installiere das Python-Paket 'tiktoken' mit: pip install tiktoken")
    st.stop()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── HARTCODED API-KEY (wie gewünscht) ─────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ─── Prompt für den Map-Schritt (Einzelantwort pro Chunk) ────────────────────
QUESTION_PROMPT_TEMPLATE = """
Du bist ein freundlicher, hilfsbereiter Assistent.
Deine Aufgabe ist es, Nutzer:innen ohne Vorwissen einfache, präzise und verständliche Antworten
auf Fragen zum Betreiberleitfaden zu geben. Vermeide Fachjargon und erkläre wenn nötig.

Nutze ausschließlich den folgenden Textauszug als Informationsquelle:
{context}

Frage:
{question}

Antwort:
"""

# ─── Prompt für den Combine-Schritt (Zusammenführung) ────────────────────────
# Verwende 'summaries' statt 'input_documents' für die Default document_variable_name
COMBINE_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Assistent.
Du bekommst mehrere kurze Antworten (Summaries) zu einer Frage und sollst sie zu einer präzisen Gesamtantwort zusammenfassen.

Frage:
{question}

Summaries:
{summaries}

Fasse sie nun verständlich und vollständig zusammen:
"""

@st.cache_resource(show_spinner=False)
def load_chain():
    # 1) Embeddings-Modell auf CPU laden (kein Meta-Tensor)
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # 2) FAISS-Index lokal laden
    vectorstore = FAISS.load_local(
        "leitfaden_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # 3) Retriever mit maximaler Marginal Relevance für diversere Treffer
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,                          # Anzahl finaler Kontexte
        "fetch_k": 20,                   # rohe Kandidaten
        "maximal_marginal_relevance": True,
        "lambda_mult": 0.5               # trade-off zwischen Relevanz und Diversität
    })

    # 4) Chat-LLM (schnelles Modell)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )

    # 5) PromptTemplates definieren
    question_prompt = PromptTemplate(
        template=QUESTION_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    combine_prompt = PromptTemplate(
        template=COMBINE_PROMPT_TEMPLATE,
        input_variables=["question", "summaries"]
    )

    # 6) RetrievalQA-Chain mit Map-Reduce
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt
        }
    )

    return chain


def main():
    # Streamlit-Konfiguration
    st.set_page_config(
        page_title="PDF Chatbot",
        layout="centered"
    )

    st.title("📘 Frag den Betreiberleitfaden")
    st.markdown(
        "💡 **Beispiel-Fragen:**\n"
        "- Was muss ich vor dem Einschalten beachten?\n"
        "- Welche Aufgaben hat der Betreiber?\n"
        "- Was steht zur Sicherheitsprüfung?"
    )

    # Formular mit Textarea und Submit-Button
    with st.form(key="frage_form", clear_on_submit=True):
        question = st.text_area(
            label="❓ Deine Frage:",
            height=150
        )
        submitted = st.form_submit_button(label="Antwort anzeigen")

    if submitted and question.strip():
        qa_chain = load_chain()
        with st.spinner("📚 Ich durchsuche den Leitfaden..."):
            try:
                answer = qa_chain.run(question)
            except Exception as e:
                st.error(f"🔴 Ein Fehler ist aufgetreten: {e}")
            else:
                if answer:
                    st.success("✅ Antwort gefunden:")
                    st.write(answer)
                else:
                    st.error("⚠️ Leider konnte ich dazu im Leitfaden nichts finden.")

if __name__ == "__main__":
    main()


