import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── Streamlit-Konfiguration ─────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Prompt-Vorlage: Überschriften + Inline-Quellen
QUESTION_PROMPT = '''
Du bist ein hilfsbereiter Assistent. Nutze ausschließlich den folgenden Textauszug als Quelle:

{context}

Frage:
{question}

Antwort:
Bitte strukturiere deine Antwort mit klaren Markdown-Überschriften (##) und füge nach jeder Aussage eine Quellenangabe in der Form (Seite X) ein.
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
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 20, "maximal_marginal_relevance": True})
    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    # PromptTemplate
    prompt = PromptTemplate(template=QUESTION_PROMPT, input_variables=["context", "question"])
    # Chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={"question_prompt": prompt, "combine_prompt": prompt},
        return_source_documents=True
    )

# Hauptfunktion
def main():
    # API-Key konfigurieren
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("🔑 API-Schlüssel fehlt. Bitte in Streamlit Secrets hinterlegen.")
        return

    st.title("📘 Frag den Betreiberleitfaden")

    # Eingabe-Feld oben
    question = st.text_input("❓ Deine Frage:")
    if st.button("🔍 Antwort anzeigen") and question.strip():
        chain = load_chain()
        with st.spinner("📚 Ich durchsuche den Leitfaden…"):
            result = chain({"query": question})
        answer = result.get("result", "")
        docs = result.get("source_documents", [])

        # Strukturierte Antwort ausgeben
        st.markdown("## ✅ Antwort")
        st.markdown(answer)

        # Quellen sammeln und anzeigen
        pages = sorted({str(doc.metadata.get("page", "?")).strip() for doc in docs})
        if pages:
            st.markdown("**Quellen:** " + ", ".join(f"Seite {p}" for p in pages))

if __name__ == "__main__":
    main()

