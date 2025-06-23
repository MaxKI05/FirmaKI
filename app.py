import os
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── Streamlit-Konfiguration ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Firmen-KI Chat", layout="wide")

# Pakete-Check
for pkg in ("tiktoken", "transformers", "sentence-transformers", "torch"):
    try:
        __import__(pkg)
    except ImportError:
        st.error(f"Bitte installiere das Paket '{pkg}' in requirements.txt.")
        st.stop()

# OpenAI-Client initialisieren
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 API-Schlüssel fehlt. Bitte in Streamlit Secrets hinterlegen.")
    st.stop()
client = OpenAI(api_key=api_key)

# Prompt-Vorlagen für strukturierte Antworten mit Inline-Quellen
QUESTION_PROMPT = '''
Du bist ein hilfsbereiter Assistent. Nutze ausschließlich den folgenden Textauszug als Quelle.

{context}

Frage:
{question}

Antwort:
Bitte strukturiere deine Antwort mit klaren Markdown-Überschriften (##) und füge nach jeder Aussage eine Quellenangabe in der Form (Seite X) ein.
'''
COMBINE_PROMPT = '''
Du bist ein hilfreicher Assistent. Du bekommst mehrere kurze Antworten (Summaries) zu einer Frage.
Fasse sie zu einer präzisen Antwort zusammen, strukturiere sie mit Markdown-Überschriften (##) und füge nach jeder Aussage (Seite X) als Quelle ein.
'''

@st.cache_resource(show_spinner=False)
def load_chain():
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    index_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(index_path):
        st.error("Index-Ordner 'leitfaden_index' nicht gefunden.")
        st.stop()
    store = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_kwargs={"k": 5, "fetch_k": 20, "maximal_marginal_relevance": True})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    q_prompt = PromptTemplate(template=QUESTION_PROMPT, input_variables=["context","question"])
    c_prompt = PromptTemplate(template=COMBINE_PROMPT, input_variables=["question","summaries"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={"question_prompt": q_prompt, "combine_prompt": c_prompt},
        return_source_documents=True
    )

# Session State initialisieren
if 'history' not in st.session_state:
    st.session_state.history = []  # List of dicts: {question, answer, followups}

# Funktion zum Absenden einer Frage
def submit_question(question_text):
    chain = load_chain()
    result = chain({"query": question_text})
    answer = result.get("result")
    docs = result.get("source_documents", [])
    entry = {"question": question_text, "answer": answer}
    # Folgefragen generieren
    fu_prompt = f"Basierend auf dieser Antwort: {answer}\nNenne drei sinnvolle Folgefragen."
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"Formuliere drei Folgefragen."},
            {"role":"user","content":fu_prompt}
        ],
        max_tokens=100
    )
    fu_text = resp.choices[0].message.content
    entry['followups'] = [line.strip('- ').strip() for line in fu_text.splitlines() if line.strip()]
    st.session_state.history.append(entry)

# Sidebar: Chatverlauf & Regenerieren
st.sidebar.header("🗨️ Chatverlauf")
for idx, entry in enumerate(st.session_state.history):
    with st.sidebar.expander(f"Frage: {entry['question']}", expanded=False):
        st.markdown(entry['answer'])
        if st.button("🔄 Regenerieren", key=f"regen_{idx}"):
            st.session_state.history = st.session_state.history[:idx]
            submit_question(entry['question'])
st.sidebar.markdown("---")
st.sidebar.markdown("**Neue Frage** im Hauptbereich unten eingeben")

# Hauptbereich: Chat-Dialog
st.markdown("# 📘 Firmen-KI Chat")
for entry in st.session_state.history:
    st.markdown(f"**Du:** {entry['question']}")
    st.markdown(f"**KI:** {entry['answer']}")
    if entry.get('followups'):
        cols = st.columns(len(entry['followups']))
        for i, fu in enumerate(entry['followups']):
            if cols[i].button(fu, key=f"fu_{i}"):
                submit_question(fu)
    st.markdown("---")

# Eingabe-Feld unten
question_text = st.text_input(
    "Deine Frage:",
    placeholder="Stelle deine Frage hier",
    key="input_field"
)
if st.button("🔍 Antwort anzeigen", key="submit_button"):
    if question_text.strip():
        submit_question(question_text)

# Entry Point
if __name__ == "__main__":
    pass
