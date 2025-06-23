import os
import streamlit as st
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── Streamlit-Konfiguration ─────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Pakete-Check
for pkg in ("tiktoken", "transformers", "sentence_transformers", "torch"):
    try:
        __import__(pkg)
    except ImportError:
        st.title("📦 Fehlendes Paket")
        st.error(f"Bitte installiere das Paket '{pkg}' in requirements.txt und redeploy die App.")
        st.stop()

# API-Key & OpenAI-Client
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.title("🔑 Kein API-Schlüssel gefunden")
    st.error("Bitte hinterlege Deinen OPENAI_API_KEY in Streamlit Secrets oder als Umgebungsvariable.")
    st.stop()
client = OpenAI(api_key=api_key)

# Prompt-Vorlagen mit finaler Antwort + Begründung + Inline-Quellen
QUESTION_PROMPT_TEMPLATE = """
Du bist ein hilfsbereiter Assistent. Nutze ausschließlich den folgenden Textauszug als Informationsquelle:

{context}

Frage:
{question}

Antwort:
Bitte gib zuerst eine klare, einzelne finale Antwort in einem Satz. 
Anschließend erkläre Schritt-für-Schritt, welche Information du auf welchen Seiten gefunden hast. 
Füge nach jeder Aussage in der Erklärung eine Quellenangabe in der Form (Seite X) ein. 
Antworte chronologisch entsprechend der Seitenreihenfolge (Seite 1 zuerst).
"""

COMBINE_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Assistent. Du erhältst mehrere kurze Antworten (Summaries) zu einer Frage.
Fasse sie zu einer einzigen finalen Antwort zusammen und gib anschließend eine nachvollziehbare Erklärung, 
geordnet nach Seiten (Seite 1 zuerst), mit Inline-Quellenangaben (Seite X) an.
"""

@st.cache_resource(show_spinner=False)
def load_chain():
    # 1) Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    # 2) FAISS-Index laden
    base_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(base_path):
        st.error(f"❌ Ordner 'leitfaden_index' nicht gefunden unter {base_path}.")
        st.stop()
    vectorstore = FAISS.load_local(
        base_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    # 3) Retriever mit größerem fetch_k
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "fetch_k": 50,
        "maximal_marginal_relevance": True,
        "lambda_mult": 0.5
    })
    # 4) LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    # 5) PromptTemplates
    question_prompt = PromptTemplate(
        template=QUESTION_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    combine_prompt = PromptTemplate(
        template=COMBINE_PROMPT_TEMPLATE,
        input_variables=["summaries", "question"]
    )
    # 6) RetrievalQA mit map_rerank
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt
        },
        return_source_documents=True
    )
    return chain

# Session-State initialisieren
if 'history' not in st.session_state:
    st.session_state.history = []  # Tuple: (question, answer, docs)

# Hauptfunktion

def main():
    st.title("📘 Frag den Betreiberleitfaden")
    question = st.text_input("❓ Deine Frage:")
    if st.button("🔍 Antwort anzeigen") and question.strip():
        chain = load_chain()
        with st.spinner("📚 Ich durchsuche den Leitfaden…"):
            res = chain({"query": question})
        answer = res.get("result", "Keine Antwort gefunden.")
        docs = res.get("source_documents", [])
        # Antwort ausgeben
        st.markdown("## ✅ Antwort")
        st.write(answer)
        # Erklärungssnippets chronologisch
        if docs:
            st.markdown("---")
            st.markdown("### Erklärung nach Seiten")
            sorted_docs = sorted(docs, key=lambda d: int(d.metadata.get("page", 0) or 0))
            for doc in sorted_docs:
                page = doc.metadata.get("page", "?")
                snippet = doc.page_content.replace("\n", " ")[:200]
                st.markdown(f"**Seite {page}:** {snippet}…")
        # Verlauf speichern
        st.session_state.history.append((question, answer, docs))

if __name__ == "__main__":
    main()
