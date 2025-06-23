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
# Kleines Logo oben rechts
logo_path = os.path.join(os.path.dirname(__file__), "assets", "ikl_logo.png")
col1, col2 = st.columns([9, 1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=50)

# Pakete-Check
for pkg in ("tiktoken", "transformers", "sentence_transformers", "torch"):
    try:
        __import__(pkg)
    except ImportError:
        st.error(f"Bitte installiere das Paket '{pkg}' in requirements.txt.")
        st.stop()

# API-Key & Client
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 API-Schlüssel fehlt. Bitte in Secrets hinterlegen.")
    st.stop()
client = OpenAI(api_key=api_key)

# Prompt-Vorlagen
QUESTION_PROMPT = '''
Du bist ein hilfsbereiter Assistent. Nutze ausschließlich den folgenden Textauszug als Quelle.

{context}

Frage:
{question}

Antwort:
Bitte strukturiere deine Antwort mit Markdown-Überschriften (##) und füge nach jeder Aussage (Seite X) ein.
'''
COMBINE_PROMPT = '''
Du bist ein hilfreicher Assistent. Du bekommst mehrere kurze Antworten (Summaries) zu einer Frage.
Fasse sie zusammen mit Überschriften (##) und Quellenangaben (Seite X).
'''

@st.cache_resource
func def load_chain():
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(path):
        st.error("Index-Ordner nicht gefunden.")
        st.stop()
    store = FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
    retr = store.as_retriever(search_kwargs={"k":5,"fetch_k":20,"maximal_marginal_relevance":True})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    q_prompt = PromptTemplate(template=QUESTION_PROMPT, input_variables=["context","question"])
    c_prompt = PromptTemplate(template=COMBINE_PROMPT, input_variables=["question","summaries"])
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retr, chain_type="map_reduce", 
        chain_type_kwargs={"question_prompt":q_prompt,"combine_prompt":c_prompt}, 
        return_source_documents=True
    )

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []  # List of dicts: {"question","answer","docs"}
if 'input' not in st.session_state:
    st.session_state.input = ''

# Haupt-UI
st.markdown("# 📘 Firmen-KI Chat")
chat_container = st.container()
with chat_container:
    for entry in st.session_state.history:
        st.markdown(f"**Du:** {entry['question']}")
        st.markdown(f"**KI:** {entry['answer']}")

# Eingabe-Feld unten

def submit_question():
    q = st.session_state.input.strip()
    if not q: return
    chain = load_chain()
    res = chain({"query": q})
    ans = res["result"]
    docs = res.get("source_documents", [])
    st.session_state.history.append({"question": q, "answer": ans, "docs": docs})
    # Follow-ups generieren
    fu_prompt = f"Basierend auf: {ans}\nNenne drei sinnvolle Folgefragen."
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"Formuliere Folgefragen."},
            {"role":"user","content":fu_prompt}
        ],
        max_tokens=100
    )
    fu_text = resp.choices[0].message.content
    fus = [line.strip('- ').strip() for line in fu_text.splitlines() if line.strip()]
    st.session_state.history[-1]["followups"] = fus
    st.session_state.input = ''

st.text_input(
    "",
    key="input",
    on_change=submit_question,
    placeholder="Stelle eine Frage...",
    label_visibility="collapsed"
)

# Follow-up Buttons
if st.session_state.history:
    last = st.session_state.history[-1]
    if 'followups' in last:
        st.markdown("---")
        cols = st.columns(len(last['followups']))
        for i, fu in enumerate(last['followups']):
            if cols[i].button(fu, key=f"fu_{i}"):
                st.session_state.input = fu
                submit_question()

