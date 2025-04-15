## === Imports e Configurações Iniciais ===
import streamlit as st

st.set_page_config(page_title="Agente de Suporte Sisand", layout="wide")

from supabase_utils import (
    salvar_mensagem,
    busca_mensagens_por_embedding,
    salvar_parametros_ia,
    carregar_parametros_ia,
)
import requests
import os
from social_utils import eh_pergunta_social, carregar_social_keywords
from dotenv import load_dotenv

load_dotenv()
import time
from PIL import Image
import pytesseract
from weaviate.util import generate_uuid5
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth
from datetime import datetime
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from openai import OpenAI
from sklearn.cluster import KMeans
import asyncio
import atexit
import weaviate
import re
from supabase import create_client, Client
from dateutil.parser import isoparse
import pandas as pd


# === Atualização: Função para salvar social_keywords de forma correta ===
def salvar_parametros_ia(chave, valor):
    supabase.table("parametros").upsert(
        [{"nome": chave, "valor": valor}], on_conflict="nome"
    ).execute()


# === Botão de teste de palavras sociais ===
if st.sidebar.button("🧪 Testar palavras sociais"):
    st.sidebar.markdown("### Resultados de teste")
    testes = [
        "Oi, tudo bem?",
        "Como faço nota fiscal?",
        "meu nome é Anderson e o seu?",
        "Você sabe emitir nota conjugada?",
    ]
    for frase in testes:
        try:
            with st.spinner(f"Consultando: {frase}"):
                r = requests.post("http://localhost:8000/ask", json={"question": frase})
                eh_social = r.json().get("sisandinho", False)
                st.sidebar.write(
                    f"'{frase}' → {'✅ Social' if eh_social else '❌ Técnica'}"
                )
        except Exception as e:
            st.sidebar.error(f"Erro com '{frase}': {e}")


# === Supabase Client ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Session state seguro ===
chaves_iniciais = {
    "modo": "chat",
    "usuario": "anonimo",
    "historico": [],
    "feedback": [],
    "feedback_aberto": [],
    "resposta_atual": "",
    "artigos_atuais": [],
    "personalidade": "🤖 Técnico direto ao ponto",
    "logado": False,
    "tickets_curadoria": [],
    "ticket_selecionado": None,
    "curadoria_pergunta": "",
    "curadoria_solucao": "",
    "curadoria_diagnostico": "",
    "curadoria_resultado": "",
    "curadoria_dica": "",
}

for chave, valor in chaves_iniciais.items():
    if chave not in st.session_state:
        st.session_state[chave] = valor


# === Parâmetros IA ===
def carregar_parametros_ia():
    dados = supabase.table("parametros").select("*").execute()
    return {d["nome"]: d["valor"] for d in dados.data}


def salvar_parametros_ia(chave, valor):
    supabase.table("parametros").upsert(
        {"nome": chave, "valor": valor}, on_conflict="nome"
    ).execute()


parametros = carregar_parametros_ia()
modelo_padrao = parametros.get("modelo", "gpt-4")
temperatura_padrao = float(parametros.get("temperatura", 0.7))
top_p_padrao = float(parametros.get("top_p", 1.0))

# === Sidebar: Configurações de IA ===
with st.sidebar.expander("⚙️ Configurações de IA", expanded=False):
    st.markdown("### 🤖 Parâmetros do Modelo")
    modelo_ia = st.selectbox(
        "Modelo de IA",
        ["gpt-3.5-turbo", "gpt-4"],
        index=["gpt-3.5-turbo", "gpt-4"].index(modelo_padrao),
    )
    temperatura = st.slider("🔥 Temperatura", 0.0, 1.0, temperatura_padrao, 0.05)
    top_p = st.slider("🎯 Top-p", 0.1, 1.0, top_p_padrao, 0.05)

    st.markdown("---")
    st.markdown("### 💬 Palavras-chave sociais")
    palavras_atuais = parametros.get("social_keywords", "")
    input_palavras = st.text_area(
        "Palavras-chave (separadas por vírgula)", value=palavras_atuais, height=100
    )

    if st.button("💾 Salvar Palavras Sociais"):
        salvar_parametros_ia("social_keywords", input_palavras)
        st.success("Palavras salvas com sucesso!")

    if st.button("💾 Salvar Parâmetros IA"):
        salvar_parametros_ia("modelo", modelo_ia)
        salvar_parametros_ia("temperatura", str(temperatura))
        salvar_parametros_ia("top_p", str(top_p))
        st.success("Parâmetros salvos com sucesso!")

# === Exibição dos parâmetros ===
with st.expander("📋 Parâmetros Atuais", expanded=True):
    st.markdown(f"**🧠 Modelo Selecionado:** `{modelo_ia}`")
    st.markdown(f"**🔥 Temperatura:** `{temperatura}`")
    st.markdown(f"**🎯 Top-p:** `{top_p}`")
    st.markdown(f"**👤 Usuário:** `{st.session_state.usuario}`")
    st.markdown(f"**💬 Personalidade:** `{st.session_state.personalidade}`")

if st.session_state.modo == "chat":
    st.success("🤖 Modo Chat Inteligente Ativo")
elif st.session_state.modo == "curadoria IA":
    st.info("🧠 Modo Curadoria IA Ativo")


# Definir menu_grupos fora da função para torná-lo acessível globalmente
menu_grupos = {
    "🔎 Busca & IA": {"🔎 Chat Inteligente": "chat"},
    "📥 Curadoria": {
        "🧠 Curadoria de Tickets": "curadoria",
        "🧩 Curadoria Manual": "curadoria",
    },
    "📊 Gestão": {
        "📊 Painel de Gestão": "painel",
        "🗂️ Conversas Finalizadas": "conversas",
        "📊 Feedbacks Recebidos": "feedbacks",
    },
    "📥 Base de Conhecimento": {
        "📥 Importar Artigos": "importar artigos",
        "📚 Ver Embeddings": "ver embeddings",
    },
    "⚙️ Configurações": {"📝 Editar Prompt do Agente": "editor_prompt"},
}

# Função para renderizar o menu de forma consolidada
def renderizar_menu():
    if "grupo_ativo" not in st.session_state:
        st.session_state.grupo_ativo = list(menu_grupos.keys())[0]
    if "modo_visualizacao" not in st.session_state:
        st.session_state.modo_visualizacao = list(
            menu_grupos[st.session_state.grupo_ativo].keys()
        )[0]

    st.sidebar.title("Navegação")
    nova_secao = st.sidebar.radio(
        "Escolha a seção:",
        list(menu_grupos.keys()),
        index=list(menu_grupos.keys()).index(st.session_state.grupo_ativo),
    )
    if nova_secao != st.session_state.grupo_ativo:
        st.session_state.grupo_ativo = nova_secao
        st.session_state.modo_visualizacao = list(menu_grupos[nova_secao].keys())[0]
        st.rerun()

    novo_modo = st.sidebar.radio(
        "Escolha o modo:",
        list(menu_grupos[st.session_state.grupo_ativo].keys()),
        index=list(menu_grupos[st.session_state.grupo_ativo].keys()).index(
            st.session_state.modo_visualizacao
        ),
    )
    if novo_modo != st.session_state.modo_visualizacao:
        st.session_state.modo_visualizacao = novo_modo
        st.rerun()

    st.session_state.modo = menu_grupos[st.session_state.grupo_ativo][
        st.session_state.modo_visualizacao
    ]


# Chamar a função para renderizar o menu
renderizar_menu()

# === Limites de processamento paralelos ===
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# === Compatibilidade com versões antigas do client ===
try:
    from weaviate.client_config import AdditionalConfig
except ModuleNotFoundError:

    class AdditionalConfig:
        def __init__(self, use_grpc=False, timeout=60):
            self.use_grpc = use_grpc
            self.timeout = 60
            self.connection = None
            self.proxies = None
            self.trust_env = True


@atexit.register
def cleanup():
    try:
        asyncio.get_event_loop().close()
    except:
        pass


load_dotenv()
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


# Função centralizada de conexão


def get_weaviate_client(skip_checks=True, use_grpc=False):
    additional_config = AdditionalConfig(use_grpc=use_grpc, timeout=60)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        skip_init_checks=skip_checks,
        additional_config=additional_config,
    )
    st.write("✅ DEBUG: client.collections.list_all()", client.collections.list_all())
    return client


# Função de embedding via OpenAI
def gerar_embedding_openai(texto):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=texto, model="text-embedding-ada-002")
    return response.data[0].embedding


# Função para obter a coleção "Article" de forma segura com debug
def get_article_collection():
    client = get_weaviate_client(skip_checks=True, use_grpc=False)
    try:
        st.write("🔍 DEBUG: client.collections:", client.collections)
        collection = client.collections.get("Article")
        st.write("🔍 DEBUG: collection.get('Article'):", collection)
        st.write("🔍 DEBUG: type(collection):", type(collection))
        if hasattr(collection, "query") and hasattr(collection.query, "near_vector"):
            return client, collection
        else:
            raise ValueError(
                "A coleção 'Article' não está configurada corretamente ou o método 'near_vector' não está disponível."
            )
    except Exception as e:
        client.close()
        raise e


# === Adição: Seleção de Modelo ===
modelos_disponiveis = ["gpt-3.5-turbo", "gpt-4"]
st.session_state.modelo_selecionado = st.sidebar.selectbox(
    "🧠 Modelo de Resposta GPT", modelos_disponiveis, index=0
)

# === CURADORIA IA ===
if st.session_state.modo == "curadoria IA":
    st.title("🧠 Curadoria IA: Busca Semântica Avançada")
    consulta = st.text_input(
        "Digite sua pergunta ou tema:", key="consulta_curadoria_ia"
    )

    modelo_escolhido = st.selectbox("🤖 Modelo de IA", ["gpt-3.5", "gpt-4"], index=0)

    if consulta:
        with st.spinner("🔎 Buscando artigos com ajuda da IA..."):
            try:
                # Debug: Exibir os parâmetros enviados para o backend
                st.write("🔍 DEBUG: Parâmetros enviados para o backend:")
                st.json({
                    "question": consulta,
                    "usuario_id": st.session_state.usuario,
                    "usuario_nome": st.session_state.usuario_nome,
                    "personalidade": "",  # não utilizado aqui
                    "use_gpt": True,
                    "limit": 10,
                    "modelo": (
                        "gpt-4" if modelo_escolhido == "gpt-4" else "gpt-3.5-turbo"
                    ),
                })

                response = requests.post(
                    "http://localhost:8000/ask",
                    json={
                        "question": consulta,
                        "usuario_id": st.session_state.usuario,
                        "usuario_nome": st.session_state.usuario_nome,
                        "personalidade": "",  # não utilizado aqui
                        "use_gpt": True,
                        "limit": 10,
                        "modelo": (
                            "gpt-4" if modelo_escolhido == "gpt-4" else "gpt-3.5-turbo"
                        ),
                    },
                    timeout=60,
                )

                # Debug: Verificar a resposta do backend
                st.write("🔍 DEBUG: Resposta do backend (status code):", response.status_code)
                if response.status_code == 200:
                    data = response.json()
                    st.write("🔍 DEBUG: Dados retornados pelo backend:")
                    st.json(data)

                    resposta = data.get("resposta", "")
                    artigos = data.get("artigos", [])
                    tempo = data.get("tempo", None)

                    if resposta:
                        st.success("✅ Resposta gerada com sucesso!")
                        st.markdown("### 💬 Resposta")
                        st.markdown(resposta)
                        if tempo:
                            st.markdown(f"**⏱️ Tempo de resposta:** {tempo} segundos")

                    if artigos:
                        st.markdown("### 📎 Fontes utilizadas")
                        for art in artigos:
                            st.markdown(
                                f"**[{art.get('title', 'Título não disponível')}]({art.get('url', '#')})** — Similaridade: **{art.get('similaridade', 'N/A')}%**"
                            )
                    else:
                        st.warning("⚠️ Nenhum artigo relevante encontrado no backend.")
                else:
                    st.error(f"Erro na API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erro ao consultar backend: {e}")


# === Login no App (colocado antes de qualquer conteúdo) ===
USUARIOS = {
    "anderson": "senha123",
    "jordan": "senha123",
    "ronan": "senha123",
    "gabriel": "senha123",
    "ivoneia": "senha123",
    "jean": "senha123",
    "janaina": "senha123",
}

# Temporário para simular nome vindo do Workspace
if "usuario_nome" not in st.session_state:
    st.session_state.usuario_nome = "Nome do usuário do Workspace"


HABILITAR_LOGIN = False  # Altere para True para ativar o login

if HABILITAR_LOGIN:
    if "logado" not in st.session_state:
        st.session_state.logado = False
    if "usuario" not in st.session_state:
        st.session_state.usuario = ""
    if not st.session_state.logado:
        st.title("🔐 Login - Agente de Suporte Sisand")
        with st.form("login_form"):
            usuario = st.text_input("Usuário")
            senha = st.text_input("Senha", type="password")
            submit = st.form_submit_button("Entrar")
        if submit:
            if USUARIOS.get(usuario) == senha:
                st.session_state.logado = True
                st.session_state.usuario = usuario
                st.success(f"Bem-vindo(a), {usuario}!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
        st.stop()

# === Configurações da Interface ===
st.title("🤖 Agente de Suporte | Sisand")
# Sidebar: Evitar textos repetidos
with st.sidebar:
    if "sidebar_rendered" not in st.session_state:
        st.sidebar.image(
            "https://www.sisand.com.br/wp-content/uploads/2022/11/logo-sisand-branco.png",
            width=180,
        )
        st.sidebar.markdown(
            """
            ### Bem-vindo!
            Digite sua dúvida e receba uma resposta baseada nos artigos de ajuda da Sisand.

            ✅ Integrado com Weaviate + GPT-4
            """
        )
        st.session_state.sidebar_rendered = True

    nova_secao = st.radio(
        "👁️ Seção",
        list(menu_grupos.keys()),
        index=list(menu_grupos.keys()).index(st.session_state.grupo_ativo),
    )
    if nova_secao != st.session_state.grupo_ativo:
        st.session_state.grupo_ativo = nova_secao
        st.session_state.modo_visualizacao = list(menu_grupos[nova_secao].keys())[0]
        st.rerun()

    nova_visualizacao = st.radio(
        "Escolha o modo",
        list(menu_grupos[st.session_state.grupo_ativo].keys()),
        index=list(menu_grupos[st.session_state.grupo_ativo].keys()).index(
            st.session_state.modo_visualizacao
        ),
    )
    if nova_visualizacao != st.session_state.modo_visualizacao:
        st.session_state.modo_visualizacao = nova_visualizacao
        st.rerun()

# Define o modo atual
st.session_state.modo = menu_grupos[st.session_state.grupo_ativo][
    st.session_state.modo_visualizacao
]


# Novo Módulo: Editor de Prompt do Agente
# Novo Módulo: Editor de Prompt do Agente (com múltiplos prompts)
if st.session_state.modo == "editor_prompt":
    st.title("📝 Editor de Prompts do Agente Inteligente")
    st.info("Aqui você pode editar os prompts usados pela IA para gerar as respostas.")

    prompts_para_editar = {
        "padrao": "🎯 Prompt Geral (Chat Inteligente)",
        "curadoria": "📚 Prompt de Curadoria (estrutura de resposta)",
    }

    for nome_prompt, descricao in prompts_para_editar.items():
        st.markdown(f"---\n### {descricao}")

        # Buscar prompt atual
        try:
            resposta = requests.get(
                "http://localhost:8000/prompt", params={"nome": nome_prompt}
            )
            if resposta.status_code == 200:
                prompt_atual = resposta.json().get("prompt", "")
            else:
                st.warning(f"⚠️ Erro ao buscar o prompt '{nome_prompt}'")
                prompt_atual = ""
        except Exception as e:
            st.error(f"Erro ao conectar com backend: {e}")
            prompt_atual = ""

        novo_prompt = st.text_area(
            f"✏️ Prompt {nome_prompt}:",
            value=prompt_atual,
            height=300,
            key=f"prompt_{nome_prompt}",
        )

        if st.button(f"💾 Salvar '{nome_prompt}'", key=f"salvar_{nome_prompt}"):
            try:
                r = requests.post(
                    "http://localhost:8000/prompt",
                    json={"nome": nome_prompt, "novo_prompt": novo_prompt},
                )
                if r.status_code == 200:
                    st.success(f"Prompt '{nome_prompt}' atualizado com sucesso! ✅")
                else:
                    st.error(f"Erro ao salvar: {r.text}")
            except Exception as e:
                st.error(f"Erro ao conectar com backend: {e}")


# Função de renderização das ações dos tickets
def renderizar_acoes_ticket(acoes):
    cores_autores = [
        ("👤", "#D6336C", "#FFF0F5"),  # Cliente
        ("🎧", "#2F80ED", "#EEF4FF"),  # Agente
    ]

    padrao = r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}) - ([^:]+):\s?(.*)"

    for acao in acoes:
        # Cabeçalho para exibir o número da ação
        st.markdown(
            f"""
            <div style='text-align:center; color:#666; margin-top:20px; margin-bottom:10px;'>
                ──────── 📝 <strong>Ação nº {acao['id']}</strong> ────────
            </div>
        """,
            unsafe_allow_html=True,
        )

        descricao = acao.get("description", "").replace("\r\n", "\n")
        linhas = descricao.split("\n")
        mensagens = []

        # Processa linhas dentro de cada ação
        for linha in linhas:
            linha = linha.strip()
            if not linha:
                continue

            match = re.match(padrao, linha)
            if match:
                data, nome_autor, mensagem = match.groups()
                mensagens.append(
                    {
                        "data": data,
                        "nome": nome_autor.strip(),
                        "conteudo": mensagem.strip(),
                    }
                )
            else:
                if mensagens:
                    mensagens[-1]["conteudo"] += "<br>" + linha
                else:
                    mensagens.append({"data": "", "nome": "Sistema", "conteudo": linha})

        ultimo_nome = None
        idx_autor = 0  # Alterna cores Cliente/Agente
        for msg in mensagens:
            if msg["nome"] == "Sistema":
                icone, cor_nome, fundo = ("⚙️", "#666666", "#F4F4F4")
            else:
                if msg["nome"] != ultimo_nome:
                    idx_autor = 0 if idx_autor == 1 else 1
                icone, cor_nome, fundo = cores_autores[idx_autor]

            autor_atual = f"{icone} {msg['nome']}"

            if msg["nome"] != ultimo_nome:
                st.markdown(
                    f"""
                    <div style="margin-top:25px; margin-bottom:5px;">
                        <span style='font-size:16px;color:{cor_nome};font-weight:bold;'>{icone} {msg['nome']}</span>
                        <span style='color:#999;font-size:12px;'> • {msg['data']}</span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style='background-color:{fundo}; padding:10px; border-radius:8px;
                            border:1px solid #DDD; font-size:15px; margin-bottom:8px;'>
                    {msg["conteudo"]}
                </div>
            """,
                unsafe_allow_html=True,
            )

            ultimo_nome = msg["nome"]


# CORREÇÃO PARA O BLOCO DE "VER EMBEDDINGS"
if st.session_state.modo == "ver embeddings":
    st.title("🧠 Embeddings dos Artigos")
    try:
        response = requests.get("http://localhost:8000/embeddings", timeout=20)
        if response.status_code != 200:
            st.error("❌ Erro ao buscar embeddings do backend.")
        else:
            artigos = response.json()  # resposta já é uma lista
            vetores = []
            titulos = []
            urls = []
            filtro = st.text_input("🔎 Filtrar por título")
            for art in artigos:
                if filtro and filtro.lower() not in art["title"].lower():
                    continue
                vetor_ok = art["vector"]
                if isinsatnce(vetor_ok, list) and all(
                    isinstance(x, (int, float)) for x in vetor_ok
                ):
                    vetores.append(vetor_ok)
                    titulos.append(art["title"])
                    urls.append(art["url"])
                else:
                    st.warning(f"⚠️ Vetor inválido para o artigo: {art['title']}")

            if len(vetores) > 1:
                if all(len(v) == len(vetores[0]) for v in vetores):
                    import pandas as pd
                    import plotly.express as px

                    vetores_np = np.array(vetores)
                    dists = np.linalg.norm(
                        vetores_np[:, None, :] - vetores_np[None, :, :], axis=-1
                    )
                    media = np.mean(dists[np.triu_indices_from(dists, k=1)])
                    st.metric("📏 Distância média entre embeddings", f"{media:.4f}")
                    st.subheader("📊 Distribuição visual (PCA)")
                    num_grupos = st.slider(
                        "Quantidade de grupos (clusters)",
                        min_value=2,
                        max_value=10,
                        value=5,
                    )
                    kmeans = KMeans(
                        n_clusters=num_grupos, random_state=42, n_init="auto"
                    )
                    labels = kmeans.fit_predict(vetores_np)
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(vetores_np)
                    df = pd.DataFrame(
                        {
                            "x": coords[:, 0],
                            "y": coords[:, 1],
                            "Título": titulos,
                            "Cluster": [f"Grupo {l+1}" for l in labels],
                            "ID": [f"{i+1}" for i in range(len(titulos))],
                        }
                    )
                    fig = px.scatter(
                        df,
                        x="x",
                        y="y",
                        color="Cluster",
                        hover_name="Título",
                        text="ID",
                        title="📊 Projeção dos embeddings em 2D por Grupo",
                        width=800,
                        height=500,
                    )
                    fig.update_traces(textposition="top center")
                    fig.update_layout(legend_title_text="Grupos de Artigos")
                    st.plotly_chart(fig)
                    st.info(
                        "*Passe o mouse sobre os pontos para ver os títulos dos artigos."
                    )
                    st.info(
                        "*Os números no gráfico correspondem à ordem dos artigos listados abaixo."
                    )
                    st.markdown("---")
                    st.subheader("🧩 Agrupamento de artigos por similaridade")
                    agrupados = {}
                    for i, label in enumerate(labels):
                        agrupados.setdefault(label, []).append((titulos[i], urls[i]))
                    for grupo, artigos_grupo in agrupados.items():
                        st.markdown(f"### 🎯 Grupo {grupo + 1}")
                        for titulo, link in artigos_grupo:
                            st.markdown(f"- [{titulo}]({link})")
                    st.markdown("---")
                    st.subheader("🔍 Artigos com maior distância do seu próprio grupo")
                    centroides = kmeans.cluster_centers_
                    dados_curadoria = []
                    for i, vetor in enumerate(vetores_np):
                        grupo_atual = labels[i]
                        centro_atual = centroides[grupo_atual]
                        dist_grupo = np.linalg.norm(vetor - centro_atual)
                        dist_outras = [
                            (g, np.linalg.norm(vetor - c))
                            for g, c in enumerate(centroides)
                            if g != grupo_atual
                        ]
                        mais_proximo = min(dist_outras, key=lambda x: x[1])
                        dados_curadoria.append(
                            {
                                "ID": i + 1,
                                "Título": titulos[i],
                                "Grupo atual": f"Grupo {grupo_atual + 1}",
                                "Distância ao grupo": round(dist_grupo, 4),
                                "Grupo mais próximo": f"Grupo {mais_proximo[0] + 1}",
                                "Distância ao mais próximo": round(mais_proximo[1], 4),
                                "Mais próximo de outro grupo?": (
                                    "✅" if mais_proximo[1] < dist_grupo else ""
                                ),
                            }
                        )
                    df_curadoria = pd.DataFrame(dados_curadoria)
                    df_curadoria = df_curadoria.sort_values(
                        by="Distância ao grupo", ascending=False
                    )
                    st.dataframe(df_curadoria, use_container_width=True)
                else:
                    st.warning(
                        "⚠️ Os vetores têm tamanhos diferentes e não podem ser comparados."
                    )
            else:
                st.info(
                    "🔍 Nenhum artigo encontrado com esse filtro ou apenas um artigo com vetor válido."
                )
    except Exception as e:
        st.error(f"Erro ao carregar embeddings: {e}")


# =====================================================================
# Módulo: Importar Artigos (Movidesk para Weaviate)
# =====================================================================
if st.session_state.modo == "importar artigos":
    st.title("📥 Importar Artigos do Movidesk para Weaviate")
    MOVI_TOKEN = os.getenv("MOVI_TOKEN")
    WEAVIATE_CLASS = "Article"
    BASE_URL_LIST = "https://api.movidesk.com/public/v1/kb/article"
    BASE_URL_SINGLE = "https://api.movidesk.com/public/v1/article"
    BASE_ARTICLE_URL = "https://sisand.movidesk.com/kb/pt-br/article"
    PAGE_SIZE = 50
    HEADERS = {"Content-Type": "application/json"}

    def from_iso_to_date(date_str):
        if not date_str:
            return None
        ds = date_str.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(ds)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

    def traduz_status(status_num):
        return {1: "Publicado", 2: "Suspenso"}.get(status_num, "N/A")

    zerar_base = st.checkbox("🧹 Zerar base antes de importar")
    importar_somente_atualizados = st.checkbox("🔁 Reimportar somente atualizados")
    if st.button("🚀 Iniciar Importação"):
        with st.spinner("Conectando ao Weaviate..."):
            client = get_weaviate_client(skip_checks=True, use_grpc=False)
            collections = client.collections.list_all()
            if zerar_base and WEAVIATE_CLASS in collections:
                client.collections.delete(WEAVIATE_CLASS)
                st.warning(f"🗑️ Coleção '{WEAVIATE_CLASS}' deletada.")
            if WEAVIATE_CLASS not in collections:
                client.collections.create(
                    name=WEAVIATE_CLASS,
                    description="Artigos do Movidesk",
                    properties=[
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="content", data_type=DataType.TEXT),
                        Property(
                            name="status",
                            data_type=DataType.TEXT,
                            skip_vectorization=True,
                        ),
                        Property(name="createdDate", data_type=DataType.DATE),
                        Property(name="updatedDate", data_type=DataType.DATE),
                        Property(
                            name="url", data_type=DataType.TEXT, skip_vectorization=True
                        ),
                    ],
                    vectorizer_config=None,
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    ),
                )
            collection = client.collections.get(WEAVIATE_CLASS)
        st.info("🔍 Buscando artigos no Movidesk...")
        all_articles = []
        page = 0
        total_importados = 0
        total_sem_conteudo = 0
        total_erros = 0
        barra = st.progress(0, text="⏳ Iniciando...")
        status_text = st.empty()
        while True:
            params = {"token": MOVI_TOKEN, "$top": PAGE_SIZE, "page": page}
            resp = requests.get(
                BASE_URL_LIST, headers=HEADERS, params=params, timeout=10
            )
            if resp.status_code != 200:
                st.error(f"⚠️ Erro ao buscar artigos: {resp.text}")
                break
            data = resp.json()
            items = data.get("items", [])
            if not items:
                break
            all_articles.extend(items)
            status_text.text(f"🔄 Página {page} com {len(items)} artigos...")
            page += 1
            time.sleep(0.5)
        total = len(all_articles)
        st.success(f"📚 Total de artigos encontrados: {total}")
        st.info("🚀 Enviando para o Weaviate...")
        for i, kb_art in enumerate(all_articles):
            barra.progress(
                (i + 1) / total, text=f"📦 Artigos importados: {i + 1}/{total}"
            )
            article_id = kb_art["id"]
            title = kb_art.get("title", "")[:60]
            single_url = f"{BASE_URL_SINGLE}/{article_id}"
            r = requests.get(single_url, headers=HEADERS, params={"token": MOVI_TOKEN})
            if r.status_code != 200:
                total_erros += 1
                st.warning(f"❌ Falha ao buscar artigo ID {article_id}")
                continue
            full = r.json()
            content = full.get("contentText", "").strip()
            if not content:
                total_sem_conteudo += 1
                st.warning(f"⚠️ Artigo sem conteúdo: {title}")
                continue
            obj = {
                "title": full.get("title", ""),
                "content": content,
                "status": traduz_status(full.get("articleStatus", 0)),
                "createdDate": from_iso_to_date(full.get("createdDate")),
                "updatedDate": from_iso_to_date(full.get("updatedDate")),
                "url": f"{BASE_ARTICLE_URL}/{article_id}/{full.get('slug', '')}",
            }
            uid = generate_uuid5(str(full["id"]))
            try:
                vetor = gerar_embedding_openai(content)
                if collection.data.exists(uuid=uid):
                    if importar_somente_atualizados:
                        db_obj = collection.data.get_by_id(uid)
                        db_updated = db_obj.properties.get("updatedDate")
                        api_updated = obj["updatedDate"]
                        if not db_updated or (api_updated and api_updated > db_updated):
                            collection.data.update(
                                uuid=uid, properties=obj, vector=vetor
                            )
                            st.info(f"♻️ Atualizado: {title}")
                            total_importados += 1
                        else:
                            continue
                    else:
                        collection.data.update(uuid=uid, properties=obj, vector=vetor)
                        st.info(f"♻️ Atualizado: {title}")
                        total_importados += 1
                else:
                    collection.data.insert(uuid=uid, properties=obj, vector=vetor)
                    st.success(f"✅ Inserido: {title}")
                    total_importados += 1
            except Exception as e:
                total_erros += 1
                st.error(f"❌ Erro no artigo '{title}': {e}")
        client.close()
        st.balloons()
        st.markdown("### ✅ Importação finalizada!")
        st.metric("Artigos inseridos/atualizados", total_importados)
        st.metric("Artigos sem conteúdo", total_sem_conteudo)
        st.metric("Erros", total_erros)

# =====================================================================
# Estado Global e Módulos: Chat, Painel e Curadoria
# =====================================================================
for chave in [
    "historico",
    "feedback",
    "feedback_aberto",
    "resposta_atual",
    "artigos_atuais",
    "modo",
    "personalidade",
]:
    if chave not in st.session_state:
        st.session_state[chave] = (
            [] if "historico" in chave or "feedback" in chave else ""
        )
st.session_state.personalidade = st.sidebar.selectbox(
    "💼 Estilo de Resposta",
    [
        "🤖 Técnico direto ao ponto",
        "🧑‍🏫 Professor explicador",
        "👩‍💼 Atendente simpática",
    ],
    index=0,
)
personalidades = {
    "🤖 Técnico direto ao ponto": "Responda de forma técnica, objetiva e com foco em eficiência.",
    "🧑‍🏫 Professor explicador": "Responda com uma linguagem didática, explicando os conceitos e motivos.",
    "👩‍💼 Atendente simpática": "Responda com simpatia, acolhimento e clareza, como uma atendente cordial.",
}


# === Módulo Chat com layout de chat moderno e sessões + OCR e feedback ===
from PIL import Image
import pytesseract
from datetime import datetime
import streamlit as st
import requests

if st.session_state.modo == "chat":
    st.markdown(
        """
        <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #1e1e1e; /* Fundo escuro */
        }
        .chat-bubble {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            max-width: 75%;
            color: #ffffff; /* Texto branco */
        }
        .user-msg {
            background-color: #007acc; /* Azul para mensagens do usuário */
            text-align: left;
        }
        .bot-msg {
            background-color: #444444; /* Cinza para mensagens do bot */
            text-align: left;
        }
        .sessao-header {
            font-weight: bold;
            margin: 20px 0 10px;
            color: #ffffff; /* Texto branco */
        }
        .tempo-msg {
            font-size: 10px;
            color: #aaaaaa; /* Texto cinza claro */
        }
        .feedback-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }
        .feedback-button {
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        .feedback-positive {
            background-color: #28a745; /* Green */
            color: white;
        }
        .feedback-negative {
            background-color: #dc3545; /* Red */
            color: white;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    if "chat_historico" not in st.session_state:
        st.session_state.chat_historico = []

    if "sessao_id_atual" not in st.session_state:
        st.session_state.sessao_id_atual = ""

    if "feedback" not in st.session_state or not isinstance(
        st.session_state.feedback, dict
    ):
        st.session_state.feedback = {}

    with st.container():
        st.markdown(
            """
            <div class='chat-container' id='chat-scroll'>
        """,
            unsafe_allow_html=True,
        )

        if st.session_state.chat_historico:
            for sessao in st.session_state.chat_historico:
                st.markdown(
                    f"<div class='sessao-header'>🗂️ Sessão {sessao['sessao_id']} - Iniciada em {sessao['inicio'][:16].replace('T', ' ')}</div>",
                    unsafe_allow_html=True,
                )
                for i, m in enumerate(sessao["mensagens"]):
                    hora_msg = m.get("data", datetime.now().isoformat())[11:16]
                    tempo = m.get("tempo", None)
                    tempo_str = f"⏱️ {tempo:.1f}s" if tempo is not None else ""
                    st.markdown(
                        f"""
                        <div class='chat-bubble user-msg'><strong>Você:</strong> <span style='float:right;'>{hora_msg}</span><br>{m['pergunta']}</div>
                        <div class='chat-bubble bot-msg'><strong>Sisandinho:</strong> <span style='float:right;'>{hora_msg} <span class='tempo-msg'>{tempo_str}</span></span><br>{m['resposta']}</div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Exibir artigos relacionados, se houver
                    artigos = m.get("artigos", [])
                    if artigos:
                        st.markdown("### 📎 Artigos relacionados:")
                        for art in artigos:
                            # Debug: Exibir os artigos processados no frontend
                            st.write("🔍 DEBUG: Artigo processado no frontend:", art)
                            similaridade = art.get("similaridade", "N/A")
                            st.markdown(
                                f"- [{art.get('title', 'Título não disponível')}]({art.get('url', '#')}) — Similaridade: **{similaridade}%**"
                            )
                    else:
                        st.warning("⚠️ Nenhum artigo relacionado encontrado para esta mensagem.")

                    # Botões de feedback
                    with st.expander("📝 Feedback desta resposta"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍", key=f"like_{sessao['sessao_id']}_{i}"):
                                st.session_state.feedback[
                                    f"{sessao['sessao_id']}_{i}"
                                ] = "positivo"
                                requests.post(
                                    "http://localhost:8000/feedback",
                                    json={
                                        "pergunta": m["pergunta"],
                                        "resposta": m["resposta"],
                                        "comentario": "",
                                        "tipo": "positivo",
                                    },
                                )
                        with col2:
                            if st.button(
                                "👎", key=f"dislike_{sessao['sessao_id']}_{i}"
                            ):
                                st.session_state.feedback[
                                    f"{sessao['sessao_id']}_{i}"
                                ] = "negativo"
                                requests.post(
                                    "http://localhost:8000/feedback",
                                    json={
                                        "pergunta": m["pergunta"],
                                        "resposta": m["resposta"],
                                        "comentario": "",
                                        "tipo": "negativo",
                                    },
                                )

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("---")
        with st.form("form_pergunta", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])  # Ajustar proporções
            with col1:
                pergunta = st.text_input("Digite sua dúvida:", key="chat_input")
            with col2:
                enviar = st.form_submit_button("Enviar")

            imagem = st.file_uploader(
                "Ou envie uma imagem (print, etc):",
                type=["png", "jpg", "jpeg"],
                key="imagem_chat",
            )

            if enviar and pergunta:
                imagem_texto = ""
                if imagem:
                    try:
                        img = Image.open(imagem)
                        imagem_texto = pytesseract.image_to_string(img)
                    except Exception as e:
                        st.warning(f"Erro ao processar imagem: {e}")

                pergunta_completa = (
                    f"{pergunta}\n\n{imagem_texto}" if imagem_texto else pergunta
                )

                with st.spinner("Gerando resposta..."):
                    try:
                        r = requests.post(
                            "http://localhost:8000/ask",
                            json={
                                "question": pergunta_completa,
                                "usuario_nome": st.session_state.usuario_nome,
                                "usuario_id": st.session_state.usuario,
                                "personalidade": personalidades[
                                    st.session_state.personalidade
                                ],
                            },
                        )
                        if r.status_code == 200:
                            data = r.json()
                            resposta = data.get("resposta", "(Sem resposta)")
                            sessao_id = data.get("sessao_id", "nova")
                            tempo_resposta = data.get("tempo", None)
                            artigos = data.get("artigos", [])
                            inicio = datetime.now().isoformat()

                            if (
                                not st.session_state.chat_historico
                                or st.session_state.chat_historico[0]["sessao_id"]
                                != sessao_id
                            ):
                                st.session_state.chat_historico.insert(
                                    0,
                                    {
                                        "sessao_id": sessao_id,
                                        "inicio": inicio,
                                        "mensagens": [],
                                    },
                                )

                            st.session_state.chat_historico[0]["mensagens"].append(
                                {
                                    "pergunta": pergunta,
                                    "resposta": resposta,
                                    "artigos": artigos,
                                    "data": datetime.now().isoformat(),
                                    "tempo": tempo_resposta,
                                }
                            )
                            st.rerun()
                        else:
                            st.error(
                                f"Erro ao buscar resposta: {r.status_code} - {r.text}"
                            )
                    except Exception as e:
                        st.error(f"Erro ao conectar com backend: {e}")


# === Novo modo: visualizar feedbacks ===
elif st.session_state.modo == "feedbacks":
    st.title("📊 Feedbacks Recebidos")
    tipo_filtro = st.selectbox(
        "Filtrar por tipo de feedback", ["Todos", "positivo", "negativo", "comentario"]
    )

    try:
        response = supabase.table("feedbacks").select("*").execute()
        feedbacks = response.data

        if tipo_filtro != "Todos":
            if tipo_filtro == "comentario":
                feedbacks = [f for f in feedbacks if f.get("comentario")]
            else:
                feedbacks = [f for f in feedbacks if f.get("tipo") == tipo_filtro]

        df = pd.DataFrame(feedbacks)
        if not df.empty:
            st.dataframe(
                df[["pergunta", "resposta", "tipo", "comentario", "created_at"]],
                use_container_width=True,
            )
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Exportar CSV", data=csv, file_name="feedbacks.csv", mime="text/csv"
            )
        else:
            st.info("Nenhum feedback encontrado para o filtro selecionado.")

    except Exception as e:
        st.error(f"Erro ao consultar Supabase: {e}")

# Módulo: Painel Administrativo
elif st.session_state.modo == "painel":
    st.subheader("📊 Painel Administrativo")
    try:
        r = requests.get("http://localhost:8000/metrics", timeout=10)
        if r.status_code == 200:
            data = r.json()
            st.metric("📨 Total de perguntas", data["total_perguntas"])
            st.metric("⏱️ Tempo médio (s)", data["tempo_medio_resposta"])
            st.metric("💬 Feedbacks abertos", data["feedbacks_recebidos"])
            st.markdown("### 🔝 Perguntas mais frequentes")
            for pergunta, count in data["perguntas_mais_frequentes"]:
                st.markdown(f"- {pergunta} ({count}x)")
            st.markdown("### 📎 Artigos mais utilizados")
            for artigo, count in data["artigos_mais_utilizados"]:
                st.markdown(f"- {artigo} ({count}x)")
        else:
            st.warning("⚠️ Falha ao buscar métricas.")
    except Exception as e:
        st.error(f"Erro ao buscar painel: {e}")
    if st.button("⬅️ Voltar para o chat"):
        st.session_state.modo = "chat"
        st.rerun()

# Módulo: Curadoria de Tickets
elif st.session_state.modo == "curadoria":
    st.title("📥 Curadoria de Tickets Resolvidos")
    st.subheader("🧹 Curadoria com apoio de IA para tickets do Movidesk")
    API_URL_TICKETS = "http://localhost:8000/movidesk-tickets"
    API_URL_WEAVIATE = "http://localhost:8000/weaviate-save"
    API_URL_CURADORIA = "http://localhost:8000/curadoria/tickets-curados"
    API_URL_GPT_CURADORIA = "http://localhost:8000/gpt-curadoria"
    API_URL_CURADORIA_LISTAR = "http://localhost:8000/curadoria"
    API_URL_ARTIGOS_WEAVIATE = "http://localhost:8000/weaviate-artigos"
    limite = st.sidebar.number_input(
        "Quantidade de tickets a buscar", min_value=1, max_value=200, value=10
    )
    if st.sidebar.button("🔄 Buscar Tickets"):
        with st.spinner("Consultando API e filtrando tickets..."):
            response = requests.get(
                API_URL_TICKETS, params={"limite": limite}, timeout=15
            )
            if response.status_code != 200:
                st.error("Erro ao buscar tickets: " + response.text)
            else:
                tickets = response.json().get("tickets", [])
                curados = []
                try:
                    curados_resp = requests.get(API_URL_CURADORIA)
                    if curados_resp.status_code == 200:
                        curados = curados_resp.json()
                except:
                    st.warning("⚠️ Não foi possível verificar os tickets já curados.")
                tickets_nao_curados = [t for t in tickets if t["id"] not in curados]
                st.session_state.tickets_curadoria = tickets_nao_curados
                st.success(
                    f"🎯 {len(tickets_nao_curados)} tickets disponíveis para curadoria."
                )
    if st.sidebar.button("📄 Ver curadorias salvas"):
        try:
            resposta = requests.get(API_URL_CURADORIA_LISTAR)
            if resposta.status_code == 200:
                lista = resposta.json()
                st.sidebar.markdown(f"**Total de curadorias:** {len(lista)}")
                for item in lista[-5:]:
                    st.sidebar.markdown(
                        f"- Ticket #{item['ticket_id']}: {item['question'][:40]}..."
                    )
            else:
                st.sidebar.warning("Não foi possível listar as curadorias.")
        except:
            st.sidebar.error("Erro ao consultar curadorias.")
    if st.sidebar.button("📚 Ver últimos artigos no Weaviate"):
        try:
            resposta = requests.get(API_URL_ARTIGOS_WEAVIATE)
            if resposta.status_code == 200:
                artigos = resposta.json()
                st.sidebar.markdown("### Últimos artigos salvos:")
                for art in artigos:
                    if art["type"] == "resposta_ticket":
                        st.sidebar.markdown(f"- {art['title']}")
            else:
                st.sidebar.warning("Não foi possível buscar os artigos.")
        except:
            st.sidebar.error("Erro ao consultar artigos no Weaviate.")
    for chave in [
        "curadoria_pergunta",
        "curadoria_solucao",
        "curadoria_diagnostico",
        "curadoria_resultado",
        "curadoria_dica",
    ]:
        if chave not in st.session_state:
            st.session_state[chave] = ""
    if "tickets_curadoria" not in st.session_state:
        st.session_state.tickets_curadoria = []
    if "ticket_selecionado" not in st.session_state:
        st.session_state.ticket_selecionado = None
    col_lista, col_detalhe = st.columns([1, 3])
    with col_lista:
        st.markdown("### 🎫 Tickets")
        for ticket in st.session_state.tickets_curadoria:
            if st.button(
                f"#{ticket['id']} - {ticket['subject']}", key=f"btn_{ticket['id']}"
            ):
                st.session_state.ticket_selecionado = ticket
    with col_detalhe:
        ticket = st.session_state.ticket_selecionado
        if ticket:
            st.markdown(f"## 🎯 Ticket #{ticket['id']} - {ticket['subject']}")
            st.write(f"📅 Criado em: {ticket['createdDate']}")
            st.write(f"📂 Categoria: {ticket.get('category', 'N/A')}")
            st.write(f"📌 Status: {ticket['status']}")
            acoes = ticket.get("actions", [])
            st.markdown("### 💬 Ações deste ticket")
            renderizar_acoes_ticket(acoes)
            st.markdown("---")
            st.subheader("📝 Preencha o conteúdo da curadoria")
            entrada_livre = st.text_area(
                "Resumo da curadoria (escreva com suas palavras)",
                value="""Pergunta do Cliente:



 Resposta do agente:
                """,
                height=200,
                key="resumo_curadoria",
            )
            if st.button("✍️ Gerar estrutura com IA") and entrada_livre.strip():
                with st.spinner("Gerando sugestão com IA..."):
                    try:
                        resposta_gpt = requests.post(
                            API_URL_GPT_CURADORIA,
                            json={
                                "texto": entrada_livre,
                                "instrucoes": "Você é um especialista em suporte técnico de sistemas ERP para concessionárias. Com base no resumo a seguir, gere uma estrutura formal com os campos: Pergunta do cliente (reescreva com clareza e tom consultivo), Solução aplicada (em passos objetivos), Diagnóstico (se possível), Resultado final (se aplicável) e uma Dica adicional útil para situações futuras.",
                            },
                        )
                        if resposta_gpt.status_code == 200:
                            dados = resposta_gpt.json()
                            st.session_state.curadoria_pergunta = dados.get(
                                "pergunta", ""
                            )
                            st.session_state.curadoria_solucao = dados.get(
                                "solucao", ""
                            )
                            st.session_state.curadoria_diagnostico = dados.get(
                                "diagnostico", ""
                            )
                            st.session_state.curadoria_resultado = dados.get(
                                "resultado", ""
                            )
                            st.session_state.curadoria_dica = dados.get("dica", "")
                        else:
                            st.warning("Falha ao gerar estrutura com IA")
                    except Exception as e:
                        st.error(f"Erro: {e}")
            pergunta = st.text_area(
                "📨 Pergunta do cliente",
                value=st.session_state.get("curadoria_pergunta", ""),
                height=100,
            )
            solucao = st.text_area(
                "✅ Solução aplicada",
                value=st.session_state.get("curadoria_solucao", ""),
                height=100,
            )
            diagnostico = st.text_area(
                "🔎 Diagnóstico (opcional)",
                value=st.session_state.get("curadoria_diagnostico", ""),
                height=80,
            )
            resultado = st.text_area(
                "📈 Resultado final (opcional)",
                value=st.session_state.get("curadoria_resultado", ""),
                height=80,
            )
            dica = st.text_area(
                "💡 Dica adicional (opcional)",
                value=st.session_state.get("curadoria_dica", ""),
                height=80,
            )
            if st.button("💾 Salvar no Weaviate", key=f"salvar_ticket_{ticket['id']}"):
                artigo = {
                    "title": f"Ticket {ticket['id']} - {ticket['subject']}",
                    "content": f"{solucao}\n\n{diagnostico}\n\n{resultado}\n\n{dica}",
                    "question": pergunta,
                    "source": f"https://sisand.com.br/ticket/{ticket['id']}",
                    "type": "resposta_ticket",
                }
                r = requests.post(API_URL_WEAVIATE, json=artigo)
                if r.status_code == 200:
                    nome_curador = st.session_state.usuario  # Usuário logado
                    curadoria_payload = {
                        "ticket_id": ticket["id"],
                        "curador": nome_curador,
                        "question": pergunta,
                        "answer": f"{solucao}\n\n{diagnostico}\n\n{resultado}\n\n{dica}".strip(),
                    }
                    r2 = requests.post(
                        "http://localhost:8000/curadoria", json=curadoria_payload
                    )
                    if r2.status_code == 200:
                        st.success(
                            "✅ Curadoria registrada e artigo salvo com sucesso!"
                        )
                        st.session_state.ticket_selecionado = None
                        st.session_state.curadoria_pergunta = ""
                        st.session_state.curadoria_solucao = ""
                        st.session_state.curadoria_diagnostico = ""
                        st.session_state.curadoria_resultado = ""
                        st.session_state.curadoria_dica = ""
                        st.rerun()
                    else:
                        st.error(
                            "❌ Salvo no Weaviate, mas falhou ao registrar curadoria."
                        )
                else:
                    st.error("❌ Erro ao salvar no Weaviate: " + r.text)


@atexit.register
def cleanup():
    try:
        asyncio.get_event_loop().close()
    except:
        pass


# === Módulo: Conversas Finalizadas ===
if st.session_state.modo == "conversas":
    st.title("🗂️ Conversas Finalizadas com o Sisandinho")
    try:
        r = requests.get(
            "http://localhost:8000/sessoes",
            params={"usuario_id": st.session_state.usuario},
        )
        if r.status_code == 200:
            sessoes = r.json().get("sessoes", [])

            if not sessoes:
                st.info("Nenhuma conversa encontrada para este usuário.")
            else:
                for idx, sessao in enumerate(sessoes):
                    inicio_formatado = isoparse(sessao["inicio"]).strftime(
                        "%d/%m/%Y %H:%M"
                    )
                    st.markdown(
                        f"---\n### 🗂️ Sessão #{idx+1} • Iniciada em: `{inicio_formatado}`"
                    )

                    for mensagem in sessao["mensagens"]:
                        horario = isoparse(mensagem["created_at"]).strftime("%H:%M")
                        st.markdown(
                            f"""
                        <div style='display:flex; flex-direction:column; margin-bottom:15px;'>
                            <div style='align-self:flex-start; background-color:#EAF4FF; padding:10px; border-radius:8px; max-width:70%;'>
                                <strong>🙋‍♂️ Você — {horario}:</strong><br>{mensagem['pergunta']}</div>
                            <div style='align-self:flex-start; background-color:#F0F0F0; padding:10px; border-radius:8px; max-width:70%; margin-top:5px;'>
                                <strong>🤖 Sisandinho — {horario}:</strong><br>{mensagem['resposta']}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        else:
            st.error(f"Erro ao buscar sessões: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"Erro ao consultar sessões: {e}")
