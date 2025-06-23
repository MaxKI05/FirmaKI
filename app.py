import os
import streamlit as st
import openai

# ─── Streamlit-Konfiguration ─────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# ─── Sicherstellen, dass alle Pakete installiert sind ─────────────────────────────────
for pkg in ("tiktoken", "transformers", "sentence_transformers", "torch"):
    try:
        __import__(pkg)
    except ImportError:
        st.title("📦 Fehlendes Paket")
        st.error(f"Bitte installiere das Paket '{pkg}' in requirements.txt und redeploy die App.")
        st.stop()

# ─── API-Key aus Secrets oder ENV-Variable ───────────────────────────────────────────────
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

# ─── Prompt-Vorlagen mit Struktur und Quellenangabe ────────────────────────────────────
QUESTION_PROMPT_TEMPLATE = """
Du bist ein hilfsbereiter Assistent. Nutze ausschließlich den folgenden Textauszug als Informationsquelle.

{context}

Frage:
{question}

Antwort:
Bitte strukturiere deine Antwort mit klaren Markdown-Überschriften (##) und füge nach jeder Aussage eine Quellenangabe in der Form (Seite X) ein.
"""

COMBINE_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Assistent. Du bekommst mehrere kurze Antworten (Summaries) zu einer Frage.

Frage:
{question}

Summaries:
{summaries}

Fasse sie zu einer präzisen Antwort zusammen und strukturiere mit Markdown-Überschriften (##). Nach jeder Aussage angebe die Quelle in der Form (Seite X).
"""

@st.cache_resource(show_spinner=False)
def load_chain():
    # Embeddings-Modell (CPU)
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    # FAISS-Index laden
    base_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(base_path):
        st.error(f"❌ Ordner 'leitfaden_index' nicht gefunden unter {base_path}.")
        st.stop()
    vectorstore = FAISS.load_local(
        base_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "maximal_marginal_relevance": True,
        "lambda_mult": 0.5
    })
    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    # PromptTemplates
    question_prompt = PromptTemplate(
        template=QUESTION_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    combine_prompt = PromptTemplate(
        template=COMBINE_PROMPT_TEMPLATE,
        input_variables=["question", "summaries"]
    )
    # RetrievalQA mit Quellen
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

# ─── Hauptfunktion ────────────────────────────────────────────────────────────────────
def main():
    st.markdown("# 📘 Frag den Betreiberleitfaden")
    st.markdown("---")

    with st.form(key="frage_form", clear_on_submit=True):
        question = st.text_area("❓ Deine Frage:", height=150)
        submitted = st.form_submit_button("🔍 Antwort anzeigen")

    if submitted and question.strip():
        chain = load_chain()
        with st.spinner("📚 Ich strukturiere und suche im Leitfaden..."):
            result = chain({"query": question})
            answer = result.get("result")
        if answer:
            st.markdown(answer)
        else:
            st.error("⚠️ Leider konnte ich dazu im Leitfaden nichts finden.")

if __name__ == "__main__":
    main()


