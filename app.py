# app.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import faiss
import numpy as np
import io
import hashlib

st.set_page_config(page_title="TCC NLP", layout="wide")
st.title("📚 Chat - Teste inicial rodando no Streamlit Cloud")

# ---------- Helpers ----------
# MELHORIA PRO PERFIL: ajuste de chunking, atual 200/50
def chunk_text(text, chunk_size=200, overlap=50):
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
    # MELHORIA PRO PERFIL: escolha do LLM
    # model_name = "google/flan-t5-small"
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 => CPU
    return pipe

embed_model = load_sentence_model()
gen_pipe = load_generation_pipeline()

# ---------- Upload PDFs ----------
uploaded = st.file_uploader("Carregue PDFs", type="pdf", accept_multiple_files=True)
if uploaded:
    # read bytes & compute a cache key (to avoid recomputing se mesmo arquivo)
    all_text = ""
    hashes = []
    for f in uploaded:
        b = f.read()
        hashes.append(file_hash_bytes(b))
        reader = PdfReader(io.BytesIO(b))
        for p in reader.pages:
            extracted = p.extract_text()
            if extracted:
                all_text += extracted + "\n"

    cache_key = "_".join(hashes)

    # ---------- Processamento de Dados (ETL: Extract, Transform, Load in Memory) ----------
    if "index_key" not in st.session_state or st.session_state["index_key"] != cache_key:
        with st.spinner("Processando e Indexando..."):
            # 1. Chunking (Bronze -> Silver)
            chunks = chunk_text(all_text, chunk_size=150, overlap=30) # Reduzido para segurança
            
            if not chunks:
                st.error("Não foi possível ler texto do PDF. Ele pode ser uma imagem escaneada.")
                st.stop()

            # 2. Embedding (Silver -> Gold/Vectors)
            embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            
            # 3. Indexing (FAISS)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            
            st.session_state["index"] = index
            st.session_state["embeddings"] = embeddings
            st.session_state["chunks"] = chunks
            st.session_state["index_key"] = cache_key
        st.success(f"Indexado com sucesso! {len(chunks)} fragmentos criados.")

    # ---------- Chat ----------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Faça uma pergunta sobre os PDFs:")

    if query:
        with st.spinner("Pensando..."):
            q_emb = embed_model.encode([query], convert_to_numpy=True)
            # MELHORIA PRO PERFIL: top-k ajustar
            # k=2 : a soma dos textos caiba nos 512 tokens do modelo.
            D, I = st.session_state["index"].search(q_emb, k=2)
            
            top_idxs = I[0].tolist()
            valid_chunks = [st.session_state["chunks"][i] for i in top_idxs if i < len(st.session_state["chunks"])]
            context = "\n\n".join(valid_chunks)
            
            # MELHORIA PRO PERFIL: Prompt Engineering
            prompt = f"Responda a pergunta com base no contexto.\n\nContexto: {context}\n\nPergunta: {query}\nResposta:"
            
            with st.expander("Ver Contexto Enviado ao Modelo"):
                st.text(prompt[:1000] + "...")

            # MUDANÇA 4: Correção dos parâmetros de geração
            # 'max_new_tokens' controla o tamanho da resposta. 
            # 'truncation=True' evita o erro de estouro, cortando o excesso se necessário.
            try:
                out = gen_pipe(
                    prompt, 
                    max_new_tokens=100, 
                    do_sample=False, 
                    truncation=True
                )
                answer = out[0]["generated_text"]
            except Exception as e:
                answer = f"Erro na geração: {str(e)}"

            st.session_state.chat_history.append((query, answer))

    # Exibir histórico
    for q, a in st.session_state.chat_history[::-1]:
        st.markdown(f"**👤 Você:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.divider()

else:
    st.info("Por favor, faça o upload dos PDFs na barra lateral ou acima.")
