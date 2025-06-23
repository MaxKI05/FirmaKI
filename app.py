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

# API-Key & OpenAI-Client
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 API-Schlüssel fehlt. Bitte in Streamlit Secrets hinterlegen.")
    st.stop()
client = OpenAI(api_key=api_key)

# Prompt-Vorlagen (headings & inline sources)
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

@st.cache_resource
# Lade und cache die RetrievalQA-Chain
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    index_path = os.path.join(os.path.dirname(__file__), "leitfaden_index")
    if not os.path.isdir(index_path):
        st.error("Index-Ordner 'leitfaden_index' nicht gefunden.")
        st.stop()
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k":5, "fetch_k":20, "maximal_marginal_relevance":True})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    q_prompt = PromptTemplate(template=QUESTION_PROMPT, input_variables=["context","question"])
    c_prompt = PromptTemplate(template=COMBINE_PROMPT, input_variables=["question","summaries"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={"question_prompt":q_prompt, "combine_prompt":c_prompt},
        return_source_documents=True
    )

# Session state initialisieren
if 'history' not in st.session_state:
    st.session_state.history = []  # List of dicts: {question, answer, followups}
if 'input' not in st.session_state:
    st.session_state.input = ''

# Haupt-UI
def main():
    st.markdown("# 📘 Firmen-KI Chat")
    st.markdown("---")

    # Chat-Verlauf
    for entry in st.session_state.history:
        st.markdown(f"**Du:** {entry['question']}")
        st.markdown(entry['answer'])
        # Follow-up Buttons (falls vorhanden)
        if 'followups' in entry and entry['followups']:
            cols = st.columns(len(entry['followups']))
            for i, fu in enumerate(entry['followups']):
                if cols[i].button(fu, key=f"fu_{entry['question']}_{i}"):
                    st.session_state.input = fu
                    submit_question()
        st.markdown("---")

    # Eingabe-Feld unten
def submit_question():
    question = st.session_state.input.strip()
    if not question:
        return
    chain = load_chain()
    with st.spinner("📚 Ich durchsuche den Leitfaden..."):
        result = chain({"query": question})
    answer = result.get("result")
    docs = result.get("source_documents", [])
    # Antwort speichern
    entry = {"question": question, "answer": answer}
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
    st.session_state.input = ''

# Eingabe-Widget mit Callback
st.text_input(
    label="Stelle eine Frage...",
    key="input",
    on_change=submit_question,
    placeholder="Gib hier deine Frage ein",
    label_visibility="collapsed"
)

if __name__ == "__main__":
    main()

