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

# Prompt-Vorlagen: prägnante finale Antwort + Erklärung + Inline-Quellen
QUESTION_PROMPT_TEMPLATE = '''
Du bist ein präziser Assistent und antwortest nur mit **einer einzigen, finalen Zahl**, wenn die Frage nach einem Wert fragt.
Anschließend liefere eine **ausführliche Erklärung** in **3–5 Sätzen**, in der du erläuterst, wie du darauf gekommen bist. Fordere die Erklärung an, die relevanten Seitenangaben (Seite X) chronologisch (Seite 1 zuerst) nennt.
Nutze nur Markdown-Überschriften, wenn wirklich nötig.

Kontext:
{context}

Frage:
{question}

Antwort:
''' 

COMBINE_PROMPT_TEMPLATE = '''
Du bist ein hilfreicher Assistent.
Fasse mehrere kurze Antworten (Summaries) zu einer Frage zu einer **eindeutigen endgültigen Antwort** zusammen.
Anschließend formuliere eine **ausführliche Erklärung** in **mindestens 3 Sätzen**, geordnet nach Seiten (Seite 1 zuerst), mit Inline-Quellen (Seite X).

Frage:
{question}

Summaries:
{summaries}

Antwort:
''' 

@st.cache_resource(show_spinner=False)
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    base_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(base_path):
        st.error(f"❌ Ordner 'leitfaden_index' nicht gefunden unter {base_path}.")
        st.stop()
    vectorstore = FAISS.load_local(base_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 50, "maximal_marginal_relevance": True})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    q_prompt = PromptTemplate(template=QUESTION_PROMPT_TEMPLATE, input_variables=["context","question"])
    c_prompt = PromptTemplate(template=COMBINE_PROMPT_TEMPLATE, input_variables=["question","summaries"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="map_reduce",
                                        chain_type_kwargs={"question_prompt": q_prompt, "combine_prompt": c_prompt},
                                        return_source_documents=True)

# Session State
if 'history' not in st.session_state:
    st.session_state.history = []

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
        # Post-Processing: Konsistente Einzelzahl
        numbers = re.findall(r"\b\d+\b", answer)
        if len(numbers) > 1:
            freq = collections.Counter(numbers)
            common, count = freq.most_common(1)[0]
            answer = f"**Eindeutige Zahl:** {common}\n\n{answer}"
        # Ausgabe
        st.markdown("## ✅ Antwort")
        st.write(answer)
        # Erklärungssnippets chronologisch
        if docs:
            st.markdown("---")
            st.markdown("### Erklärung nach Seiten")
            for doc in sorted(docs, key=lambda d: int(d.metadata.get("page", 0) or 0)):
                page = doc.metadata.get("page")
                if str(page).isdigit():
                    snippet = doc.page_content.replace("\n"," ")[:200]
                    st.markdown(f"**Seite {page}:** {snippet}…")
        st.session_state.history.append((question, answer, docs))

if __name__ == "__main__":
    main()

