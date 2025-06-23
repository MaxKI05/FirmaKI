import os
import sys
import streamlit as st
import openai

# ─── UI-Konfiguration & Logo laden ───────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide")
# Header mit Firmenlogo
st.image(
    os.path.join(os.path.dirname(__file__), "assets", "ikl_logo.png"),
    width=200,
    caption=""
)

# ─── Sicherstellen, dass alle Pakete installiert sind (sonst freundliche Fehlermeldung) ───
for pkg in ("tiktoken", "transformers", "sentence_transformers", "torch"):
    try:
        __import__(pkg)
    except ImportError:
        st.title("📦 Fehlendes Paket")
        st.error(f"Bitte installiere das Paket '{pkg}' in requirements.txt und redeploy die App.")
        st.stop()

# ─── API-Key aus Secrets oder ENV-Variable ──────────────────────────────────────────────
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.title("🔑 Kein API-Schlüssel gefunden")
    st.error("Bitte hinterlege Deinen OPENAI_API_KEY in Streamlit Secrets oder als Umgebungsvariable.")
    st.stop()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── Prompt-Vorlagen ─────────────────────────────────────────────────────────────────────
QUESTION_PROMPT_TEMPLATE = """
Du bist ein freundlicher, hilfsbereiter Assistent.
Nutze ausschließlich den folgenden Textauszug als Informationsquelle:

{context}

Frage:
{question}

Antwort:
"""

COMBINE_PROMPT_TEMPLATE = """
Du bist ein hilfsbereiter Assistent.
Du bekommst mehrere kurze Antworten (Summaries) zu einer Frage und sollst sie zu einer Gesamtantwort zusammenfassen.

Frage:
{question}

Summaries:
{summaries}

Fasse sie nun verständlich und vollständig zusammen:
"""

@st.cache_resource(show_spinner=False)
def load_chain():
    # 1) Embeddings auf CPU
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # 2) FAISS-Index laden (Index-Ordner muss im Repo liegen)
    base_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(base_path):
        st.error(f"❌ Ordner 'leitfaden_index' nicht gefunden unter {base_path}.")
        st.stop()

    vectorstore = FAISS.load_local(
        base_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # 3) Retriever
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "maximal_marginal_relevance": True,
        "lambda_mult": 0.5
    })

    # 4) LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )

    # 5) Prompts
    question_prompt = PromptTemplate(
        template=QUESTION_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    combine_prompt = PromptTemplate(
        template=COMBINE_PROMPT_TEMPLATE,
        input_variables=["question", "summaries"]
    )

    # 6) RetrievalQA mit Quellen zurückgeben
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


def main():
    st.markdown("# 📘 Frag den Betreiberleitfaden")
    st.markdown("---")

    with st.sidebar:
        st.header("Einstellungen")
        temp = st.slider("Temperature", 0.0, 1.0, 0.0)
        st.write("Ladezeit optimiert für präzise Antworten.")

    with st.form(key="frage_form", clear_on_submit=True):
        question = st.text_area("❓ Deine Frage:", height=150)
        submitted = st.form_submit_button("🔍 Antwort anzeigen")

    if submitted and question.strip():
        qa_chain = load_chain()
        with st.spinner("📚 Ich durchsuche den Leitfaden..."):
            res = qa_chain({"query": question})
            answer = res.get("result")
            docs = res.get("source_documents", [])

        if answer:
            # Überschrift für die Antwort
            st.markdown("## ✅ Antwort")
            # Markdown-Styling: Absätze und Listen
            st.markdown(answer)

            # Quellenabschnitt mit Seitenzahlen
            pages = sorted({str(doc.metadata.get("page", "?")).strip() for doc in docs})
            if pages:
                st.markdown("---")
                st.markdown("#### Quellen")
                for p in pages:
                    st.markdown(f"- Seite {p}")
        else:
            st.error("⚠️ Leider konnte ich dazu im Leitfaden nichts finden.")

    # Footer mit Logo und Copyright
    st.markdown("---")
    st.image(
        os.path.join(os.path.dirname(__file__), "assets", "ikl_logo.png"),
        width=100
    )
    st.markdown("© 2025 IKL GmbH | Version 1.0")

if __name__ == "__main__":
    main()



