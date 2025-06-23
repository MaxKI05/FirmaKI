import os
import streamlit as st
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── Streamlit-Konfiguration ─────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Prompt-Vorlagen
QUESTION_PROMPT = '''
Du bist ein hilfsbereiter Assistent. Nutze ausschließlich den folgenden Textauszug als Quelle:

{context}

Frage:
{question}

Antwort:
Beantworte klar und strukturiert, verwende Markdown-Überschriften nur wenn es sinnvoll ist, und füge nach jeder Aussage eine Quellenangabe in der Form (Seite X) ein. Antworte chronologisch gemäß der Seitenreihenfolge (Seite 1 zuerst).
''' 
COMBINE_PROMPT = '''
Du bist ein hilfreicher Assistent. Du bekommst mehrere kurze Antworten (Summaries) zu einer Frage.
Fasse sie in der korrekten Reihenfolge zusammen (erst Seite 1, zuletzt Seite 50), strukturiere nur wenn nötig mit Markdown-Überschriften, und füge nach jeder Aussage (Seite X) als Quelle ein.
'''

@st.cache_resource(show_spinner=False)
def load_chain():
    """Lädt und cached die RetrievalQA-Chain mit FAISS."""
    # Embeddings auf CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    # FAISS-Index laden
    index_dir = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(index_dir):
        st.error(f"Index-Ordner nicht gefunden: {index_dir}")
        st.stop()
    store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_kwargs={"k": 5, "fetch_k": 20, "maximal_marginal_relevance": True})
    # LLM konfigurieren
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    # PromptTemplates
    question_template = PromptTemplate(template=QUESTION_PROMPT, input_variables=["context", "question"])
    combine_template = PromptTemplate(template=COMBINE_PROMPT, input_variables=["summaries", "question"])
    # Chain erstellen
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={"question_prompt": question_template, "combine_prompt": combine_template},
        return_source_documents=True
    )

# Hauptfunktion
def main():
    # API-Key setzen
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🔑 API-Schlüssel fehlt. Bitte in Streamlit Secrets hinterlegen.")
        return
    openai.api_key = api_key

    st.title("📘 Frag den Betreiberleitfaden")
    # Eingabe-Feld oben
    question = st.text_input("❓ Deine Frage:")
    if st.button("🔍 Antwort anzeigen") and question.strip():
        chain = load_chain()
        with st.spinner("📚 Ich durchsuche den Leitfaden…"):
            result = chain({"query": question})
        answer = result.get("result", "")
        docs = result.get("source_documents", [])

        # Antwort ausgeben
        st.markdown("## ✅ Antwort")
        st.markdown(answer)
        # Quellen anzeigen
        pages = sorted({str(doc.metadata.get("page", "?")).strip() for doc in docs})
        if pages:
            st.markdown("**Quellen:** " + ", ".join(f"Seite {p}" for p in pages))

if __name__ == "__main__":
    main()
