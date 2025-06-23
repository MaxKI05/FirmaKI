import os
import streamlit as st
import openai

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

# API-Key
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

# Prompt-Vorlagen
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

Fasse sie zu einer präzisen Antwort zusammen und strukturiere mit Markdown-Überschriften (##). Nach jeder Aussage gebe die Quelle in der Form (Seite X) an.
"""

@st.cache_resource(show_spinner=False)
def load_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    base_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(base_path):
        st.error(f"❌ Ordner 'leitfaden_index' nicht gefunden unter {base_path}.")
        st.stop()
    vectorstore = FAISS.load_local(
        base_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "maximal_marginal_relevance": True,
        "lambda_mult": 0.5
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
    st.session_state.history = []

# Hauptfunktion

def main():
    st.sidebar.header("🗨️ Chatverlauf")
    for idx, (q, a, docs) in enumerate(st.session_state.history):
        with st.sidebar.expander(f"Frage: {q}", expanded=False):
            st.markdown(a)
            if st.button("🔄 Regenerieren", key=f"regen_{idx}"):
                st.session_state.history = st.session_state.history[:idx]
                st.session_state.current = q
                st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("Stelle eine neue Frage im Hauptbereich.")

    st.markdown("# 📘 Frag den Betreiberleitfaden")
    st.markdown("---")

    with st.form(key="frage_form", clear_on_submit=True):
        question = st.text_input("❓ Deine Frage:")
        submitted = st.form_submit_button("🔍 Antwort anzeigen")

    if submitted and question:
        chain = load_chain()
        # Direkter Aufruf ohne Streaming
        with st.spinner("📚 Suche im Leitfaden und generiere Antwort..."):
            res = chain({"query": question})
            answer = res.get("result")
            docs = res.get("source_documents", [])
        # Anzeige der Antwort
        st.markdown(answer)
        # Inline-Quellen (Seitenzahlen nach Aussagen bereits im Antworttext)
        # Chatverlauf speichern
        st.session_state.history.append((question, answer, docs))
        # Folgefragen
        followup_prompt = (
            f"Basierend auf dieser Antwort: {answer}\n" 
            "Nenne drei sinnvolle Folgefragen für den Nutzer.")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"Du sollst Folgefragen liefern."},
                      {"role":"user","content":followup_prompt}],
            max_tokens=100
        )
        followup_text = response.choices[0].message.content
        st.markdown("---")
        st.markdown("### Folgefragen")
        for line in followup_text.split("\n"):
            if line.strip():
                clean = line.lstrip("- 123.").strip()
                if st.button(clean, key=f"fu_{clean}"):
                    st.session_state.current = clean
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
