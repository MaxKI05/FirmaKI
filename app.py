import os
import re
import collections
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

# Prompt-Vorlagen: finale Antwort + ausführliche Erklärung + Inline-Quellen
QUESTION_PROMPT_TEMPLATE = '''
Du bist ein präziser Assistent und gibst bei Mengenfragen nur **eine finale Zahl** an.
Anschließend lieferst du eine **ausführliche Erklärung** in **3–5 Sätzen**, in der du beschreibst, wie du darauf gekommen bist.
Führe die relevanten Seiten chronologisch (Seite 1 zuerst) auf und füge nach jeder Aussage eine Quellenangabe in der Form (Seite X) ein.

Kontext:
{context}

Frage:
{question}

Antwort:
'''

COMBINE_PROMPT_TEMPLATE = '''
Du bist ein hilfreicher Assistent.
Fasse mehrere kurze Antworten (Summaries) zu einer Frage zu einer **eindeutigen Antwort** zusammen.
Formuliere danach eine **ausführliche Erklärung** in mindestens 3 Sätzen, geordnet nach Seiten (Seite 1 zuerst), mit Inline-Quellen (Seite X).

Frage:
{question}

Summaries:
{summaries}

Antwort:
'''

@st.cache_resource(show_spinner=False)
def load_chain():
    # Embeddings und FAISS laden
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    index_dir = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(index_dir):
        st.error(f"❌ Ordner 'leitfaden_index' nicht gefunden: {index_dir}")
        st.stop()
    store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_kwargs={"k":5, "fetch_k":50, "maximal_marginal_relevance":True})
    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    # Prompts
    q_prompt = PromptTemplate(template=QUESTION_PROMPT_TEMPLATE, input_variables=["context","question"])
    c_prompt = PromptTemplate(template=COMBINE_PROMPT_TEMPLATE, input_variables=["question","summaries"])
    # Chain erstellen
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={"question_prompt":q_prompt, "combine_prompt":c_prompt},
        return_source_documents=True
    )

# Session State für Verlauf
if 'history' not in st.session_state:
    st.session_state.history = []  # List of tuples: (question, answer, docs)

# Funktion, um eine Antwort zu generieren

def generate_answer(query: str):
    chain = load_chain()
    result = chain({"query": query})
    answer = result.get("result", "Keine Antwort gefunden.")
    docs = result.get("source_documents", [])
    # Konsistente Einzelzahl bei mehrfachen Zahlen
    nums = re.findall(r"\b\d+\b", answer)
    if len(nums) > 1:
        freq = collections.Counter(nums)
        common, _ = freq.most_common(1)[0]
        answer = f"**Eindeutige Zahl:** {common}\n\n{answer}"
    return answer, docs

# Hauptfunktion

def main():
    # Sidebar: Chat-Verlauf & Regenerieren
    st.sidebar.header("🗨️ Chatverlauf")
    for i, (q, a, docs) in enumerate(st.session_state.history):
        with st.sidebar.expander(f"Frage: {q}", expanded=False):
            st.markdown(a)
            if st.button("🔄 Regenerieren", key=f"regen_{i}"):
                # Nur diese Frage neu beantworten
                new_a, new_docs = generate_answer(q)
                st.session_state.history[i] = (q, new_a, new_docs)
                st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("Stelle eine neue Frage im Hauptbereich.")

    # Hauptbereich: Eingabe & Ausgabe
    st.title("📘 Frag den Betreiberleitfaden")
    question = st.text_input("❓ Deine Frage:", key="input_field")
    if st.button("🔍 Antwort anzeigen") and question.strip():
        ans, docs = generate_answer(question)
        # Antwort ausgeben
        st.markdown("## ✅ Antwort")
        st.write(ans)
        # Erklärungssnippets chronologisch
        if docs:
            st.markdown("---")
            st.markdown("### Erklärung nach Seiten")
            for d in sorted(docs, key=lambda x: int(x.metadata.get("page", 0) or 0)):
                pg = d.metadata.get("page")
                if str(pg).isdigit():
                    snippet = d.page_content.replace("\n"," ")[:200]
                    st.markdown(f"**Seite {pg}:** {snippet}…")
        # Verlauf speichern
        st.session_state.history.append((question, ans, docs))

if __name__ == "__main__":
    main()
