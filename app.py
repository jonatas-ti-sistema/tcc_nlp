# app.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import faiss
import numpy as np
import io, hashlib

st.set_page_config(page_title="RAG PDF Chat (offline)", layout="wide")
st.title("📚 RAG Chat - 2 PDFs (100% grátis, roda no Streamlit Cloud)")

# ---------- Helpers ----------
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def file_hash_bytes(b: bytes):
    import hashlib
    return hashlib.md5(b).hexdigest()

# ---------- Load models (cached across reruns while container vive) ----------
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_generation_pipeline():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 => CPU
    return pipe

embed_model = load_sentence_model()
gen_pipe = load_generation_pipeline()

# ---------- Upload PDFs ----------
uploaded = st.file_uploader("Carregue até 2 PDFs", type="pdf", accept_multiple_files=True)
if uploaded:
    # read bytes & compute a cache key (to avoid recomputing se mesmo arquivo)
    all_text = ""
    hashes = []
    for f in uploaded:
        b = f.read()
        hashes.append(file_hash_bytes(b))
        reader = PdfReader(io.BytesIO(b))
        for p in reader.pages:
            all_text += (p.extract_text() or "") + "\n"

    cache_key = "_".join(hashes)

    # Build chunks + embeddings + FAISS index (em memória)
    if "index_key" not in st.session_state or st.session_state["index_key"] != cache_key:
        with st.spinner("Indexando PDFs (criando embeddings)..."):
            chunks = chunk_text(all_text, chunk_size=400, overlap=80)
            embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            # store in session_state
            st.session_state["index"] = index
            st.session_state["embeddings"] = embeddings
            st.session_state["chunks"] = chunks
            st.session_state["index_key"] = cache_key
        st.success("Index pronto ✅")

    st.write(f"Chunks criados: {len(st.session_state['chunks'])}")

    # ---------- Chat ----------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Faça uma pergunta sobre os PDFs:")

    if query:
        with st.spinner("Buscando nos PDFs e gerando resposta..."):
            q_emb = embed_model.encode([query], convert_to_numpy=True)
            D, I = st.session_state["index"].search(q_emb, k=4)  # top 4
            top_idxs = I[0].tolist()
            context = "\n\n".join([st.session_state["chunks"][i] for i in top_idxs if i < len(st.session_state["chunks"])])
            prompt = f"Use apenas o contexto abaixo para responder a pergunta.\n\nContexto:\n{context}\n\nPergunta: {query}\nResposta (seja conciso):"
            out = gen_pipe(prompt, max_length=256, do_sample=False)
            answer = out[0]["generated_text"]
            st.session_state.chat_history.append((query, answer))

    # show history
    for q, a in st.session_state.chat_history[::-1]:
        st.markdown(f"**Você:** {q}")
        st.markdown(f"**Bot:** {a}")
else:
    st.info("Carregue 1 ou 2 PDFs para começar. Recomendo usar PDFs curtos (ex.: até algumas dezenas de páginas).")
