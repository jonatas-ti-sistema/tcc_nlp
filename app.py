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
    return hashlib.md5(b).hexdigest()

# ---------- 2. Load Models (Cache) ----------
@st.cache_resource
def load_models():
    # Carrega Embeddings
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # MELHORIA PRO PERFIL: escolha do LLM
    # model_name = "google/flan-t5-small"
    model_name = "google/flan-t5-base" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1) 
    return embed_model, pipe

embed_model, gen_pipe = load_models()

# ---------- 3. Sidebar & Upload ----------
with st.sidebar:
    st.header("📂 Gestão de Arquivos")
    uploaded = st.file_uploader("Carregue seus PDFs aqui", type="pdf", accept_multiple_files=True)
index_ready = False

if uploaded:
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

    # ---------- 4. Processamento (Indexação) ----------
    if "index_key" not in st.session_state or st.session_state["index_key"] != cache_key:
        with st.spinner("⚙️ Processando PDFs e criando memória vetorial..."):
            chunks = chunk_text(all_text, chunk_size=200, overlap=50)
            
            if chunks:
                # Cria embeddings
                embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
                
                # Cria Index FAISS
                dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(embeddings)
                
                # Salva no Session State
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["index_key"] = cache_key
                st.success(f"Concluído! {len(chunks)} trechos de texto processados.")
            else:
                st.error("Erro: Não foi possível extrair texto dos PDFs.")
    index_ready = True

# ---------- 5. Interface Chat ----------

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# A. Exibe as mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# B. Campo de entrada do Chat
if index_ready:
    if prompt_user := st.chat_input("Faça sua pergunta sobre os documentos..."):
        
        with st.chat_message("user"):
            st.markdown(prompt_user)
        # Salva no histórico
        st.session_state.messages.append({"role": "user", "content": prompt_user})

        # Resposta
        with st.chat_message("assistant"):
            with st.spinner("Lendo documentos e formulando resposta..."):
                
                # --- Lógica de Busca (Retrieval) ---
                q_emb = embed_model.encode([prompt_user], convert_to_numpy=True)
                # MELHORIA PRO PERFIL: top-k ajustar
                # k=3 : soma textos caiba 512 tokens / pra dar um pouco mais de contexto
                D, I = st.session_state["index"].search(q_emb, k=3)
                top_idxs = I[0].tolist()
                
                # Recupera os textos
                context_chunks = [st.session_state["chunks"][i] for i in top_idxs if i < len(st.session_state["chunks"])]
                context_text = "\n\n".join(context_chunks)
                
                # MELHORIA PRO PERFIL: escolha melhoria do prompt
                # --- Engenharia de Prompt ---
                final_prompt = f"Baseado APENAS no contexto abaixo, responda a pergunta.\n\nContexto:\n{context_text}\n\nPergunta: {prompt_user}\nResposta:"

                try:
                    output = gen_pipe(
                        final_prompt, 
                        max_new_tokens=300,
                        do_sample=False,     # Deterministico (sempre a mesma resposta para a mesma pergunta)
                        truncation=True
                    )
                    response_text = output[0]["generated_text"]
                except Exception as e:
                    response_text = f"Desculpe, tive um erro interno: {str(e)}"
                
                st.markdown(response_text)
                
                # (Opcional) Expander para ver o que o robô leu (Debugging)
                with st.expander("🕵️‍♂️ Ver trechos usados do PDF"):
                    st.write(context_text)

        # Salva resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    st.info("👈 Comece carregando um PDF na barra lateral.")