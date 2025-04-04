## === Imports e Configurações Iniciais ===
import streamlit as st
st.set_page_config(page_title="Agente de Suporte Sisand", layout="wide")

import requests
import time
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from weaviate import connect_to_wcs
from weaviate.auth import AuthApiKey
from weaviate.util import generate_uuid5
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from datetime import datetime
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from openai import OpenAI
from sklearn.cluster import KMeans
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # ou ajuste conforme sua máquina
import asyncio
import atexit
from weaviate.classes.query import MetadataQuery


@atexit.register
def cleanup():
    try:
        asyncio.get_event_loop().close()
    except:
        pass

load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# === Login no App (colocado antes de qualquer conteúdo) ===
USUARIOS = {
    "anderson": "senha123",
    "jordan": "senha123",
    "ronan": "senha123",
    "gabriel": "senha123",
    "ivoneia": "senha123",
    "jean": "senha123",
    "janaina": "senha123"
}

# === Controle de Login ===
HABILITAR_LOGIN = False  # Altere para True quando quiser reativar o login

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

st.sidebar.image("https://www.sisand.com.br/wp-content/uploads/2022/11/logo-sisand-branco.png", width=180)
st.sidebar.markdown("""
### Bem-vindo!
Digite sua dúvida e receba uma resposta baseada nos artigos de ajuda da Sisand.

✅ Integrado com Weaviate + GPT-4
""")

# === Menu de Visualização ===
# === Menu Lateral Organizado com Destaque do Modo Ativo ===
# === Menu Inteligente com Seção Ativa ===

# Dicionários de cada grupo
menu_grupos = {
    "🔎 Busca & IA": {
        "🔎 Chat Inteligente": "chat",
        "🧠 Curadoria IA": "curadoria IA"
    },
    "📊 Gestão": {
        "📊 Painel de Gestão": "painel",
        "🧩 Curadoria Manual": "curadoria"
    },
    "📥 Base de Conhecimento": {
        "📥 Importar Artigos": "importar artigos",
        "📚 Ver Embeddings": "ver embeddings"
    }
}

# Selecionar grupo ativo
grupo_ativo = st.sidebar.radio("👁️ Seção", list(menu_grupos.keys()), key="grupo_ativo")

# Mostrar os modos da seção ativa
rotulos = list(menu_grupos[grupo_ativo].keys())
modo_selecionado_label = st.sidebar.radio("Escolha o modo", rotulos, key="modo_visualizacao")

# Atualizar o modo da sessão com o valor interno
st.session_state.modo = menu_grupos[grupo_ativo][modo_selecionado_label]


# === Função para gerar embeddings via OpenAI ===
def gerar_embedding_openai(texto):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# === Continuação do código... (mantém os blocos subsequentes existentes) ===



# === Modo Ver Embeddings ===
if st.session_state.modo == "ver embeddings":
    st.title("🧠 Embeddings dos Artigos")

    try:
        client = connect_to_wcs(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )

        try:
            collection = client.collections.get("Article")
            artigos = collection.query.fetch_objects(limit=100, include_vector=True)

            vetores = []
            titulos = []
            urls = []

            filtro = st.text_input("🔎 Filtrar por título")

            for art in artigos.objects:
                if filtro and filtro.lower() not in art.properties['title'].lower():
                    continue

                vetor_ok = art.vector
                if isinstance(vetor_ok, dict):
                    vetor_ok = vetor_ok.get("default", [])

                if isinstance(vetor_ok, list) and all(isinstance(x, (int, float)) for x in vetor_ok):
                    vetores.append(vetor_ok)
                    titulos.append(art.properties['title'])
                    urls.append(art.properties['url'])
                else:
                    st.warning(f"⚠️ Vetor inválido para o artigo: {art.properties['title']}")

            if len(vetores) > 1:
                if all(len(v) == len(vetores[0]) for v in vetores):
                    import numpy as np
                    import pandas as pd
                    import plotly.express as px
                    from sklearn.decomposition import PCA
                    from sklearn.cluster import KMeans

                    vetores_np = np.array(vetores)

                    # Distância média entre embeddings
                    dists = np.linalg.norm(vetores_np[:, None, :] - vetores_np[None, :, :], axis=-1)
                    media = np.mean(dists[np.triu_indices_from(dists, k=1)])
                    st.metric("📏 Distância média entre embeddings", f"{media:.4f}")

                    st.subheader("📊 Distribuição visual (PCA)")
                    num_grupos = st.slider("Quantidade de grupos (clusters)", min_value=2, max_value=10, value=5)
                    kmeans = KMeans(n_clusters=num_grupos, random_state=42, n_init='auto')
                    labels = kmeans.fit_predict(vetores_np)

                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(vetores_np)

                    df = pd.DataFrame({
                        'x': coords[:, 0],
                        'y': coords[:, 1],
                        'Título': titulos,
                        'Cluster': [f"Grupo {l+1}" for l in labels],
                        'ID': [f"{i+1}" for i in range(len(titulos))]
                    })

                    fig = px.scatter(
                        df,
                        x='x',
                        y='y',
                        color='Cluster',
                        hover_name='Título',
                        text='ID',
                        title="📊 Projeção dos embeddings em 2D por Grupo",
                        width=800,
                        height=500
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(legend_title_text='Grupos de Artigos')

                    st.plotly_chart(fig)
                    st.info("*Passe o mouse sobre os pontos para ver os títulos dos artigos.")
                    st.info("*Os números no gráfico correspondem à ordem dos artigos listados abaixo.")

                    st.markdown("---")
                    st.subheader("🧩 Agrupamento de artigos por similaridade")
                    agrupados = {}
                    for i, label in enumerate(labels):
                        agrupados.setdefault(label, []).append((titulos[i], urls[i]))

                    for grupo, artigos_grupo in agrupados.items():
                        st.markdown(f"### 🎯 Grupo {grupo + 1}")
                        for titulo, link in artigos_grupo:
                            st.markdown(f"- [{titulo}]({link})")

                    # CURADORIA: análise de distância ao centróide
                    st.markdown("---")
                    st.subheader("🔍 Artigos com maior distância do seu próprio grupo")

                    centroides = kmeans.cluster_centers_
                    dados_curadoria = []

                    for i, vetor in enumerate(vetores_np):
                        grupo_atual = labels[i]
                        centro_atual = centroides[grupo_atual]
                        dist_grupo = np.linalg.norm(vetor - centro_atual)

                        dist_outras = [(g, np.linalg.norm(vetor - c)) for g, c in enumerate(centroides) if g != grupo_atual]
                        mais_proximo = min(dist_outras, key=lambda x: x[1])

                        dados_curadoria.append({
                            "ID": i + 1,
                            "Título": titulos[i],
                            "Grupo atual": f"Grupo {grupo_atual + 1}",
                            "Distância ao grupo": round(dist_grupo, 4),
                            "Grupo mais próximo": f"Grupo {mais_proximo[0] + 1}",
                            "Distância ao mais próximo": round(mais_proximo[1], 4),
                            "Mais próximo de outro grupo?": "✅" if mais_proximo[1] < dist_grupo else "",
                            # "Link": urls[i]  # Ative se quiser exibir os links
                        })

                    df_curadoria = pd.DataFrame(dados_curadoria)
                    df_curadoria = df_curadoria.sort_values(by="Distância ao grupo", ascending=False)
                    st.dataframe(df_curadoria, use_container_width=True)

                else:
                    st.warning("⚠️ Os vetores têm tamanhos diferentes e não podem ser comparados.")
            else:
                st.info("🔍 Nenhum artigo encontrado com esse filtro ou apenas um artigo com vetor válido.")

        finally:
            client.close()

    except Exception as e:
        st.error(f"Erro ao carregar embeddings: {e}")

if st.session_state.modo == "curadoria IA":
    st.title("🧠 Curadoria IA: Busca Semântica Avançada")

    consulta = st.text_input("Digite sua pergunta ou tema:")

    if consulta:
        try:
            import weaviate  # Certifique-se de ter instalado a versão 4.11.3 ou superior
            import os
            from weaviate.auth import AuthApiKey

            # Conectar ao Weaviate diretamente com weaviate.Client
            client = weaviate.Client(
                url=os.getenv("WEAVIATE_URL"),
                auth_client_secret=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
            )

            # Gerar embedding da consulta (usando o próprio Weaviate)
            embed_consulta = gerar_embedding_openai(consulta)
            st.write(embed_consulta)

            # Buscar top 20 artigos mais próximos utilizando a query GraphQL
            resultados = client.query.get("Article", ["title", "url", "content"])\
                .with_near_vector({"vector": embed_consulta})\
                .with_limit(20)\
                .with_additional(["distance"])\
                .do()


            artigos_candidatos = []
            for obj in resultados.objects:
                vetor = obj.vector
                titulo = obj.properties.get('title', 'Sem título')
                url = obj.properties.get('url', '')
                conteudo = obj.properties.get('content', '')  # deve existir no schema

                artigos_candidatos.append({
                    'vetor': vetor,
                    'titulo': titulo,
                    'url': url,
                    'conteudo': conteudo
                })

            # Re-ranking com cosine_similarity
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            vetores_candidatos = np.array([a['vetor'] for a in artigos_candidatos])
            sim_scores = cosine_similarity(np.array(embed_consulta).reshape(1, -1), vetores_candidatos)[0]

            artigos_ordenados = sorted(
                zip(artigos_candidatos, sim_scores),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            # Exibir resultados
            st.markdown("## 🎯 Resultados mais relevantes")

            for artigo, score in artigos_ordenados:
                st.markdown(f"**{artigo['titulo']}**  \n🔗 [Acessar artigo]({artigo['url']})  \n🧠 Similaridade: `{score:.4f}`")

                if artigo['conteudo']:
                    trecho = artigo['conteudo'][:300]
                    st.markdown(f"> {trecho}...")

                st.markdown("---")

            client.close()

        except Exception as e:
            st.error(f"Erro ao executar a busca: {e}")


# === Modo Importar Artigos ===
if st.session_state.modo == "importar artigos":
    st.title("📥 Importar Artigos do Movidesk para Weaviate")

    MOVI_TOKEN = os.getenv("MOVI_TOKEN")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
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
            client = connect_to_wcs(
                cluster_url=WEAVIATE_URL,
                auth_credentials=AuthApiKey(WEAVIATE_API_KEY)
            )
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
                        Property(name="status", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="createdDate", data_type=DataType.DATE),
                        Property(name="updatedDate", data_type=DataType.DATE),
                        Property(name="url", data_type=DataType.TEXT, skip_vectorization=True),
                    ],
                    vectorizer_config=None,
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    )
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
            resp = requests.get(BASE_URL_LIST, headers=HEADERS, params=params, timeout=10)
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
            barra.progress((i + 1) / total, text=f"📦 Artigos importados: {i + 1}/{total}")
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
                "url": f"{BASE_ARTICLE_URL}/{article_id}/{full.get('slug', '')}"
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
                            collection.data.update(uuid=uid, properties=obj, vector=vetor)
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


# Os modos chat, painel e curadoria continuam abaixo...
# Devem ser incluídos na sequência com elif como já estavam organizados antes.

# === Estado Global ===
for chave in ["historico", "feedback", "feedback_aberto", "resposta_atual", "artigos_atuais", "modo", "personalidade"]:
    if chave not in st.session_state:
        st.session_state[chave] = [] if "historico" in chave or "feedback" in chave else ""

st.session_state.personalidade = st.sidebar.selectbox(
    "💼 Estilo de Resposta",
    ["🤖 Técnico direto ao ponto", "🧑‍🏫 Professor explicador", "👩‍💼 Atendente simpática"],
    index=0
)

personalidades = {
    "🤖 Técnico direto ao ponto": "Responda de forma técnica, objetiva e com foco em eficiência.",
    "🧑‍🏫 Professor explicador": "Responda com uma linguagem didática, explicando os conceitos e motivos.",
    "👩‍💼 Atendente simpática": "Responda com simpatia, acolhimento e clareza, como uma atendente cordial."
}

# === Modo Chat ===
if st.session_state.modo == "chat":
    with st.expander("💡 Sugestões de perguntas"):
        st.markdown("""
        - Como emitir uma nota fiscal?
        - O que significa a mensagem de inadimplência?
        - Como configurar uma operação interna?
        - Como faço o inventário?
        """)

    with st.form("form_pergunta"):
        pergunta = st.text_input("Digite sua dúvida:", placeholder="Ex: Como emitir uma nota fiscal?")
        imagem = st.file_uploader("Enviar imagem ou print (opcional)", type=["png", "jpg", "jpeg"])
        enviar = st.form_submit_button("📩 Enviar")

    if enviar and pergunta:
        imagem_texto = ""
        if imagem:
            try:
                img = Image.open(imagem)
                imagem_texto = pytesseract.image_to_string(img)
            except Exception as e:
                st.warning(f"Erro ao processar a imagem: {e}")

        with st.spinner("Consultando artigos e gerando resposta..."):
            inicio = time.time()
            try:
                full_question = f"{pergunta}\n\n{imagem_texto}" if imagem_texto else pergunta
                response = requests.post("http://localhost:8000/ask", json={
                    "question": full_question,
                    "personalidade": personalidades[st.session_state.personalidade]
                }, timeout=60)

                if response.status_code == 200:
                    data = response.json()
                    resposta = data.get("resposta", "(Sem resposta)")
                    artigos = data.get("artigos", [])
                    tempo = round(time.time() - inicio, 2)

                    st.session_state.resposta_atual = resposta
                    st.session_state.artigos_atuais = artigos

                    st.success("✅ Resposta recebida!")
                    st.markdown(f"**⏱️ Tempo de resposta:** {tempo} segundos")
                    st.markdown("---")
                    st.markdown("### 💬 Resposta")
                    st.markdown(resposta)

                    if artigos:
                        st.markdown("### 📎 Fontes utilizadas")
                        for art in artigos:
                            st.markdown(f"**[{art['title']}]({art['url']})** 🔗")
                            with st.expander("🔍 Visualizar trecho consultado"):
                                st.markdown(f"> {art['snippet']}")

                    st.session_state.historico.insert(0, {
                        "pergunta": pergunta,
                        "resposta": resposta,
                        "tempo": tempo
                    })
                else:
                    st.error("Erro na API: " + response.text)
            except Exception as e:
                st.error(f"Erro ao conectar com o backend: {e}")

    if st.session_state.resposta_atual:
        col3, col4 = st.columns(2)

        with col3:
            if st.button("🧠 Resumir resposta"):
                with st.spinner("Gerando resumo..."):
                    resumo = requests.post("http://localhost:8000/ask", json={
                        "question": f"Resuma em tópicos:\n\n{st.session_state.resposta_atual}",
                        "personalidade": personalidades[st.session_state.personalidade]
                    }, timeout=60)
                    if resumo.status_code == 200:
                        st.markdown("### 📝 Resumo")
                        st.markdown(resumo.json().get("resposta", ""))

        with col4:
            if st.button("😄 Explicar de outro jeito"):
                with st.spinner("Gerando nova explicação..."):
                    nova = requests.post("http://localhost:8000/ask", json={
                        "question": f"Explique com linguagem simples:\n\n{st.session_state.resposta_atual}",
                        "personalidade": personalidades[st.session_state.personalidade]
                    }, timeout=60)
                    if nova.status_code == 200:
                        st.markdown("### 😄 Explicação alternativa")
                        st.markdown(nova.json().get("resposta", ""))

        feedback_texto = st.text_area("Essa resposta te ajudou? Deseja deixar um comentário?")
        if st.button("💾 Enviar comentário"):
            requests.post("http://localhost:8000/feedback", json={
                "pergunta": pergunta,
                "resposta": st.session_state.resposta_atual,
                "comentario": feedback_texto.strip()
            })
            st.success("Comentário salvo com sucesso! ✅")

    if st.session_state.historico:
        st.markdown("---")
        st.subheader("📜 Histórico")
        for idx, h in enumerate(st.session_state.historico):
            with st.expander(f"{idx+1}. {h['pergunta']}"):
                st.markdown(h["resposta"])
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍", key=f"like_{idx}"):
                        st.session_state.feedback[idx] = "positivo"
                with col2:
                    if st.button("👎", key=f"dislike_{idx}"):
                        st.session_state.feedback[idx] = "negativo"
                if idx in st.session_state.feedback:
                    st.markdown(f"**Feedback:** {st.session_state.feedback[idx].capitalize()}")

# === Modo Painel ===
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


# === CONTINUAÇÃO DO ARQUIVO app_streamlit.py ===
# (modo curadoria com melhoria usando IA para gerar conteúdo estruturado)

# ... (mantém todo o conteúdo acima do modo curadoria)

# === Modo Curadoria ===
elif st.session_state.modo == "curadoria":
    st.title("📥 Curadoria de Tickets Resolvidos")
    st.subheader("🧹 Curadoria com apoio de IA para tickets do Movidesk")

    API_URL_TICKETS = "http://localhost:8000/movidesk-tickets"
    API_URL_WEAVIATE = "http://localhost:8000/weaviate-save"
    API_URL_CURADORIA = "http://localhost:8000/curadoria/tickets-curados"
    API_URL_GPT_CURADORIA = "http://localhost:8000/gpt-curadoria"
    API_URL_CURADORIA_LISTAR = "http://localhost:8000/curadoria"
    API_URL_ARTIGOS_WEAVIATE = "http://localhost:8000/weaviate-artigos"

    limite = st.sidebar.number_input("Quantidade de tickets a buscar", min_value=1, max_value=200, value=10)

    if st.sidebar.button("🔄 Buscar Tickets"):
        with st.spinner("Consultando API e filtrando tickets..."):
            response = requests.get(API_URL_TICKETS, params={"limite": limite}, timeout=15)
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

                tickets_nao_curados = [t for t in tickets if t['id'] not in curados]
                st.session_state.tickets_curadoria = tickets_nao_curados
                st.success(f"🎯 {len(tickets_nao_curados)} tickets disponíveis para curadoria.")

    if st.sidebar.button("📄 Ver curadorias salvas"):
        try:
            resposta = requests.get(API_URL_CURADORIA_LISTAR)
            if resposta.status_code == 200:
                lista = resposta.json()
                st.sidebar.markdown(f"**Total de curadorias:** {len(lista)}")
                for item in lista[-5:]:
                    st.sidebar.markdown(f"- Ticket #{item['ticket_id']}: {item['question'][:40]}...")
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

    for chave in ["curadoria_pergunta", "curadoria_solucao", "curadoria_diagnostico", "curadoria_resultado", "curadoria_dica"]:
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
            if st.button(f"#{ticket['id']} - {ticket['subject']}", key=f"btn_{ticket['id']}"):
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
                key="resumo_curadoria"
            )
            if st.button("✍️ Gerar estrutura com IA") and entrada_livre.strip():
                with st.spinner("Gerando sugestão com IA..."):
                    try:
                        resposta_gpt = requests.post(API_URL_GPT_CURADORIA, json={
                            "texto": entrada_livre,
                            "instrucoes": "Você é um especialista em suporte técnico de sistemas ERP para concessionárias. Com base no resumo a seguir, gere uma estrutura formal com os campos: Pergunta do cliente (reescreva com clareza e tom consultivo), Solução aplicada (em passos objetivos), Diagnóstico (se possível), Resultado final (se aplicável) e uma Dica adicional útil para situações futuras."
                        })
                        if resposta_gpt.status_code == 200:
                            dados = resposta_gpt.json()
                            st.session_state.curadoria_pergunta = dados.get("pergunta", "")
                            st.session_state.curadoria_solucao = dados.get("solucao", "")
                            st.session_state.curadoria_diagnostico = dados.get("diagnostico", "")
                            st.session_state.curadoria_resultado = dados.get("resultado", "")
                            st.session_state.curadoria_dica = dados.get("dica", "")
                        else:
                            st.warning("Falha ao gerar estrutura com IA")
                    except Exception as e:
                        st.error(f"Erro: {e}")

            pergunta = st.text_area("📨 Pergunta do cliente", value=st.session_state.get("curadoria_pergunta", ""), height=100)
            solucao = st.text_area("✅ Solução aplicada", value=st.session_state.get("curadoria_solucao", ""), height=100)
            diagnostico = st.text_area("🔎 Diagnóstico (opcional)", value=st.session_state.get("curadoria_diagnostico", ""), height=80)
            resultado = st.text_area("📈 Resultado final (opcional)", value=st.session_state.get("curadoria_resultado", ""), height=80)
            dica = st.text_area("💡 Dica adicional (opcional)", value=st.session_state.get("curadoria_dica", ""), height=80)

            if st.button("💾 Salvar no Weaviate", key=f"salvar_ticket_{ticket['id']}"):
                artigo = {
                    "title": f"Ticket {ticket['id']} - {ticket['subject']}",
                    "content": f"{solucao}\n\n{diagnostico}\n\n{resultado}\n\n{dica}",
                    "question": pergunta,
                    "source": f"https://sisand.com.br/ticket/{ticket['id']}",
                    "type": "resposta_ticket"
                }
                r = requests.post(API_URL_WEAVIATE, json=artigo)
                if r.status_code == 200:
                    nome_curador = st.session_state.usuario  # Substituído pelo usuário logado
                    curadoria_payload = {
                        "ticket_id": ticket['id'],
                        "curador": nome_curador,
                        "question": pergunta,
                        "answer": f"{solucao}\n\n{diagnostico}\n\n{resultado}\n\n{dica}".strip()
                    }
                    r2 = requests.post("http://localhost:8000/curadoria", json=curadoria_payload)
                    if r2.status_code == 200:
                        st.success("✅ Curadoria registrada e artigo salvo com sucesso!")
                        st.session_state.ticket_selecionado = None
                        st.session_state.curadoria_pergunta = ""
                        st.session_state.curadoria_solucao = ""
                        st.session_state.curadoria_diagnostico = ""
                        st.session_state.curadoria_resultado = ""
                        st.session_state.curadoria_dica = ""
                        st.rerun()
                    else:
                        st.error("❌ Salvo no Weaviate, mas falhou ao registrar curadoria.")
                else:
                    st.error("❌ Erro ao salvar no Weaviate: " + r.text)

import asyncio
import atexit

@atexit.register
def cleanup():
    try:
        asyncio.get_event_loop().close()
    except:
        pass
