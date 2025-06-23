from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# PDF laden
loader = PyMuPDFLoader("Betreiberleitfaden_avMR.pdf")
documents = loader.load()

# In Chunks teilen
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Kostenloses Embedding-Modell laden
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vektorindex erstellen
db = FAISS.from_documents(chunks, embedding_model)
db.save_local("leitfaden_index")

print("✅ Kostenloser Vektorindex wurde erfolgreich erstellt!")

