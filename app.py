import os
import io
import time
import faiss
import hashlib
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime
from github import Github, GithubException


st.set_page_config(page_title="TCC NLP", layout="wide")
st.title("📚 Chat - Teste inicial rodando no Streamlit Cloud")

LOG_FILE = "chat_log.csv"

chunk_size=100
overlap=50

def log_interaction_github(question, response, context, time_taken):
    # 1. Recuperar as credenciais dos Secrets
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    file_path = st.secrets["FILE_PATH"]

    try:
        # 2. Conectar ao GitHub
        g = Github(token)
        repo = g.get_repo(repo_name)

        # 3. Preparar a nova linha
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Limpeza simples para não quebrar a estrutura do Pipe "|"
        clean_question = question.replace("\n", " ").replace("\r", "").replace("|", "")
        clean_response = response.replace("\n", " ").replace("\r", "").replace("|", "")
        clean_context = context.replace("\n", " ").replace("\r", "").replace("|", "")

        new_line = f"{timestamp}|{clean_question}|{clean_response}|{clean_context}|{time_taken:.2f}"

        # 4. Tentar pegar o arquivo existente e atualizar
        try:
            contents = repo.get_contents(file_path)
            csv_content = contents.decoded_content.decode("utf-8")

            # Garante que haja uma quebra de linha antes de adicionar o novo registro
            if csv_content and not csv_content.endswith("\n"):
                csv_content += "\n"

            updated_content = csv_content + new_line

            repo.update_file(
                path=contents.path,
                message=f"Log update: {timestamp}",
                content=updated_content,
                sha=contents.sha,
            )
            st.toast("Log salvo no GitHub com sucesso!", icon="☁️")

        except GithubException as e:
            # Se o arquivo não for encontrado (404), cria ele
            if e.status == 404:
                header = "Timestamp|Pergunta|Resposta|Contexto_Usado|Tempo_Segundos\n"
                create_content = header + new_line
                repo.create_file(
                    path=file_path,
                    message=f"Create log file: {timestamp}",
                    content=create_content,
                )
                st.toast("Arquivo criado e salvo no GitHub!", icon="✨")
            else:
                raise e

    except Exception as e:
        st.error(f"Erro ao salvar no GitHub: {e}")


# ---------- Helpers ----------
# MELHORIA PRO PERFIL: ajuste de chunking, atual 100/50
def chunk_text(text, chunk_size=100, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def file_hash_bytes(b: bytes):
    return hashlib.md5(b).hexdigest()


# ---------- 2. Load Models (Cache) ----------
@st.cache_resource
def load_models():
    # Carrega Embeddings
    # melhoria pro perfil: escolha do modelo de embedding
    # embed_model = SentenceTransformer("all-MiniLM-L6-v2")
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
    uploaded = st.file_uploader(
        "Carregue seus PDFs aqui", type="pdf", accept_multiple_files=True
    )
    st.sidebar.markdown("---")
    st.sidebar.info("Os logs estão sendo salvos automaticamente no GitHub.")
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
            chunks = chunk_text(all_text, chunk_size, overlap)

            if chunks:
                # Cria embeddings
                embeddings = embed_model.encode(
                    chunks, convert_to_numpy=True, show_progress_bar=False
                )

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
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if index_ready:
    if prompt_user := st.chat_input("Faça sua pergunta sobre os documentos..."):
        with st.chat_message("user"):
            st.markdown(prompt_user)
        st.session_state.messages.append({"role": "user", "content": prompt_user})

        with st.chat_message("assistant"):
            with st.spinner("Lendo documentos e formulando resposta..."):
                # --- INICIO DA CONTAGEM DE TEMPO ---
                start_time = time.time()

                q_emb = embed_model.encode([prompt_user], convert_to_numpy=True)
                D, Idx = st.session_state["index"].search(q_emb, k=6)
                top_idxs = Idx[0].tolist()

                context_chunks = [
                    st.session_state["chunks"][i]
                    for i in top_idxs
                    if i < len(st.session_state["chunks"])
                ]
                context_text = "\n\n".join(context_chunks)

                final_prompt = f"Baseado APENAS no contexto abaixo, responda a pergunta.\n\nContexto:\n{context_text}\n\nPergunta: {prompt_user}\nResposta:"

                try:
                    output = gen_pipe(
                        final_prompt,
                        max_new_tokens=500,
                        do_sample=False,
                        truncation=True,
                    )
                    response_text = output[0]["generated_text"]
                except Exception as e:
                    response_text = f"Desculpe, tive um erro interno: {str(e)}"

                # --- FIM DA CONTAGEM DE TEMPO ---
                end_time = time.time()
                elapsed_time = end_time - start_time

                # --- SALVAR LOG NO GITHUB ---
                with st.spinner("Salvando registro na nuvem..."):
                    log_interaction_github(
                        prompt_user, response_text, context_text, elapsed_time
                    )
                st.toast("Log salvo com sucesso!", icon="💾")

                st.markdown(response_text)

                with st.expander(f"🕵️‍♂️ Ver trechos usados (Tempo: {elapsed_time:.2f}s)"):
                    st.write(context_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

else:
    st.info("👈 Comece carregando um PDF na barra lateral.")
