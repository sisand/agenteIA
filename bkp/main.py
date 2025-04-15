from dotenv import load_dotenv
load_dotenv()

# 📦 Imports padrão
import os
import json
import time
import traceback
import requests
from datetime import datetime, timedelta
from collections import Counter
from contextlib import asynccontextmanager
import uuid

# 🧠 Tipagem
from typing import Optional, List

# 🚀 Bibliotecas externas
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.query import MetadataQuery
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import isoparse

# 🧹 Integrações
from supabase_utils import salvar_mensagem, busca_mensagens_por_embedding
from supabase import create_client
from social_utils import eh_pergunta_social, gerar_resposta_social
from ask_langchain import router as langchain_router
from app.clients import connect_clients, close_clients, get_supabase_client, get_openai_client, get_weaviate_client

from app.langchain_chain import criar_qa_chain

# 📊 Métricas em memória
perguntas_registradas = []
artigos_utilizados = Counter()
feedbacks_abertos = []
tempos_resposta = []

# 🧾 Modelos de Dados
class AskRequest(BaseModel):
    question: str
    personalidade: str = ""
    use_gpt: bool = True
    limit: int = 3
    modelo: Optional[str] = None
    usuario_login: Optional[str] = "anonimo"
    usuario_nome: Optional[str] = None
    historico: Optional[List[dict]] = []

class Curadoria(BaseModel):
    ticket_id: int
    curador: str
    question: str
    answer: str


class FeedbackAberto(BaseModel):
    pergunta: str
    resposta: str
    comentario: str
    tipo: Optional[str] = None
    usuario_id: Optional[int] = None  # Changed to Optional[int]

class ArticleInput(BaseModel):
    title: str
    content: str
    question: str = ""
    source: str = ""
    type: str = "resposta_ticket"

class PromptAtualizacao(BaseModel):
    nome: str
    conteudo: str

# 🔁 Ciclo de vida da aplicação
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        connect_clients()
        print("✅ Weaviate conectado:", get_weaviate_client())
        criar_qa_chain()
        yield
    except asyncio.CancelledError:
        print("⚠️ Aplicação interrompida manualmente.")
    finally:
        close_clients()
        print("🔒 Recursos liberados com sucesso.")

from app.domain.chat import router as chat_router

app = FastAPI(lifespan=lifespan)

# Registrar rotas
app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(langchain_router, prefix="/ask_langchain")


@app.post("/ask")
def ask_question(payload: AskRequest):
    try:
        print(f"📥 Recebendo pergunta: {payload.question}")
        supabase = get_supabase_client()
        openai_client = get_openai_client()

        question = payload.question.strip()
        personalidade = payload.personalidade.strip()
        use_gpt = payload.use_gpt
        limit = payload.limit
        modelo = payload.modelo or "gpt-3.5-turbo"
        usuario_login = payload.usuario_login or "anonimo"
        nome_usuario = payload.usuario_nome or usuario_login or "amigo"

        # Verificar se é uma pergunta social
        if eh_pergunta_social(question):
            resposta = gerar_resposta_social(question, nome_usuario)
            print(f"🤝 Pergunta social detectada. Resposta: {resposta}")
            return {"resposta": resposta, "artigos": [], "tempo": 0.1, "sisandinho": True}

        # Gerar embedding e buscar artigos semanticamente
        print("🔍 Buscando artigos semanticamente...")
        resultado_busca = buscar_artigos_semanticamente(
            pergunta=question,
            personalidade=personalidade,
            usar_gpt=use_gpt,
            limite=limit,
        )

        # Processar resultado da busca
        resposta = resultado_busca.get("resposta", "")
        artigos = resultado_busca.get("artigos", [])
        tempo = resultado_busca.get("tempo", 0)

        # Salvar mensagem
        usuario = supabase.table("usuarios").select("id").eq("login", usuario_login).limit(1).execute()
        usuario_id = usuario.data[0]["id"] if usuario.data else None
        if usuario_id:
            sessao_id = obter_ou_criar_sessao(supabase, usuario_id)
            salvar_mensagem(usuario_id, question, resposta, None, sessao_id)

        return {
            "resposta": resposta,
            "artigos": artigos,
            "tempo": tempo,
            "sisandinho": False,
        }

    except Exception as e:
        print(f"❌ Erro no endpoint /ask: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    total_perguntas = len(perguntas_registradas)
    perguntas_frequentes = Counter(perguntas_registradas).most_common(5)
    artigos_mais_usados = artigos_utilizados.most_common(5)
    media_tempo = round(sum(tempos_resposta) / len(tempos_resposta), 2) if tempos_resposta else 0
    return {
        "total_perguntas": total_perguntas,
        "tempo_medio_resposta": media_tempo,
        "perguntas_mais_frequentes": perguntas_frequentes,
        "artigos_mais_utilizados": artigos_mais_usados,
        "feedbacks_recebidos": len(feedbacks_abertos),
    }

@app.post("/feedback")
async def salvar_feedback(feedback: FeedbackAberto):
    try:
        feedback_data = feedback.dict()
        feedback_data["criado_em"] = get_brazil_time().isoformat()
        
        # Ensure usuario_id is an integer from the database
        if not feedback_data.get("usuario_id"):
            # Get usuario_id from usuarios table if not provided
            usuario = get_supabase_client().table("usuarios").select("id").eq("login", feedback.usuario_login).limit(1).execute()
            if usuario.data:
                feedback_data["usuario_id"] = usuario.data[0]["id"]
            else:
                feedback_data["usuario_id"] = None

        get_supabase_client().table("feedbacks").insert(feedback_data).execute()
        feedbacks_abertos.append(feedback_data)
        return {"status": "Comentário salvo com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def carregar_parametros_ia():
    dados = get_supabase_client().table("parametros").select("nome", "valor").execute().data
    return {item["nome"]: item["valor"] for item in dados}


def salvar_parametros_ia(modelo, temperatura, top_p):
    get_supabase_client().table("parametros").upsert(
        [
            {"nome": "modelo", "valor": modelo},
            {"nome": "temperatura", "valor": str(temperatura)},
            {"nome": "top_p", "valor": str(top_p)},
        ]
    ).execute()


CURADORIA_FILE = "curadorias.json"


# Endpoint para retornar embeddings dos artigos
@app.get("/embeddings")
async def listar_embeddings(limit: int = 100):
    collection = get_weaviate_client.collections.get("Article")
    artigos = collection.query.fetch_objects(limit=limit, include_vector=True)
    resposta = []
    for art in artigos.objects:
        vetor = art.vector
        if isinstance(vetor, dict):
            vetor = vetor.get("default", [])
        if not vetor or not isinstance(vetor, list):
            continue
        resposta.append(
            {
                "title": art.properties.get("title", "Sem Título"),
                "url": art.properties.get("url", "#"),
                "vector": vetor,
            }
        )
    return resposta


@app.get("/curadoria")
def listar_curadorias() -> List[dict]:
    if not os.path.exists(CURADORIA_FILE):
        return []
    with open(CURADORIA_FILE, "r") as f:
        return json.load(f)


@app.get("/curadoria/tickets-curados")
def listar_tickets_curados() -> List[int]:
    if not os.path.exists(CURADORIA_FILE):
        return []
    with open(CURADORIA_FILE, "r") as f:
        curadorias = json.load(f)
    return [c["ticket_id"] for c in curadorias]


@app.post("/gpt-curadoria")
async def gerar_curadoria_com_gpt(payload: dict = Body(...)):
    try:
        texto = payload.get("texto", "").strip()
        if not texto:
            raise ValueError("Texto não informado.")

        # Buscar o prompt salvo com nome=curadoria
        print("🔍 Buscando prompt para curadoria...")
        prompt_curadoria = buscar_prompt("curadoria")
        print(f"✅ Prompt encontrado: {prompt_curadoria}")

        # Substituir o texto no prompt
        prompt_completo = prompt_curadoria.replace("{texto}", texto)

        # Gerar resposta com GPT usando o prompt no system
        print("🤖 Gerando curadoria com GPT...")
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt_completo},  # Prompt salvo no system
            ],
        )

        texto_gerado = completion.choices[0].message.content.strip()
        print(f"✅ Resposta gerada: {texto_gerado}")

        # Parse do JSON gerado
        import json
        dados = json.loads(texto_gerado)
        return dados

    except Exception as e:
        print(f"❌ Erro no endpoint /gpt-curadoria: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/curadoria")
def registrar_curadoria(curadoria: Curadoria):
    curadoria_data = {
        "ticket_id": curadoria.ticket_id,
        "curador": curadoria.curador,
        "question": curadoria.question,
        "answer": curadoria.answer,
        "data": datetime.now().isoformat(),
    }

    if not os.path.exists(CURADORIA_FILE):
        with open(CURADORIA_FILE, "w") as f:
            json.dump([], f)

    with open(CURADORIA_FILE, "r") as f:
        curadorias = json.load(f)

    if any(c["ticket_id"] == curadoria.ticket_id for c in curadorias):
        raise HTTPException(status_code=400, detail="Este ticket já foi curado.")

    curadorias.append(curadoria_data)
    with open(CURADORIA_FILE, "w") as f:
        json.dump(curadorias, f, indent=2)

    return {"message": "Curadoria registrada com sucesso ✅"}


def gerar_embedding_openai(texto: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=texto, model="text-embedding-ada-002")
    return response.data[0].embedding


# --- FASTAPI: FUNÇÃO PARA BUSCAR PROMPT ---


def buscar_prompt(nome: str = "padrao") -> str:
    try:
        result = (
            get_supabase_client().table("prompts")
            .select("conteudo")
            .eq("nome", nome)
            .eq("ativo", True)
            .limit(1)
            .execute()
        )

        if result.data:
            return result.data[0]["conteudo"]
    except Exception as e:
        print("Erro ao buscar prompt:", e)

    return "Prompt não encontrado."


def buscar_artigos_semanticamente(pergunta, personalidade="", usar_gpt=True, limite=3):
    try:
        # Obter cliente Weaviate
        client = get_weaviate_client()

        # Gerar embedding para a pergunta
        query_vector = gerar_embedding_openai(pergunta)

        # Buscar artigos relacionados no Weaviate
        results = client.collections.get("Article").query.near_vector(
            near_vector=query_vector, 
            limit=limite, 
            return_properties=["title", "url", "content"]
        )
        artigos = results.objects

        if not artigos:
            return {"resposta": "Nenhum artigo relevante encontrado.", "artigos": []}

        # Processar artigos encontrados
        artigos_resumidos = []
        for artigo in artigos:
            titulo = artigo.properties.get("title", "")
            url = artigo.properties.get("url", "#")
            snippet = artigo.properties.get("content", "")[:300] + "..."
            artigos_resumidos.append({
                "title": titulo,
                "url": url,
                "snippet": snippet
            })
            print(f"🔍 DEBUG: Artigo processado: {titulo}")
        # Gerar resposta com GPT, se configurado
        if usar_gpt:
            import time

            # Obter cliente OpenAI
            openai_client = get_openai_client()

            inicio = time.time()

            prompt = (
                f"{personalidade}\n\n"
                "Abaixo estão alguns artigos técnicos sobre o sistema da Sisand:\n\n"
                + "\n\n".join([f"{a['title']}: {a['url']}" for a in artigos_resumidos])
                + "\n\nCom base nos artigos acima, elabore uma resposta profissional e estruturada para a seguinte pergunta:\n\n"
                f"Pergunta: {pergunta}\n\n"
                "Forneça sua resposta seguindo estas diretrizes:\n"
                "1. Comece com uma visão geral concisa do tema\n"
                "2. Apresente os pontos principais de forma clara e organizada\n"
                "3. Use formatação markdown quando necessário para destacar informações importantes\n"
                "4. Se houver passos a seguir, enumere-os claramente\n"
                "5. Conclua com uma síntese prática\n\n"
                "Resposta:\n"
            )
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo" if len(prompt) > 3000 else "gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um especialista técnico em sistemas ERP, com profundo conhecimento do sistema Sisand. "
                            "Suas respostas devem ser:\n"
                            "- Profissionais e objetivas\n"
                            "- Tecnicamente precisas\n"
                            "- Bem estruturadas e organizadas\n"
                            "- Práticas e orientadas à solução\n"
                            "Use formatação markdown quando necessário para melhorar a clareza."
                        )
                    },
                    {"role": "user", "content": prompt.strip()},
                ],
            )
            fim = time.time()
            tempo_resposta = round(fim - inicio, 2)
            tempos_resposta.append(tempo_resposta)
            perguntas_registradas.append(pergunta)

            return {
                "resposta": completion.choices[0].message.content.strip(),
                "artigos": artigos_resumidos,
                "tempo": tempo_resposta,
            }

        return {"resposta": None, "artigos": artigos_resumidos}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-only")
async def buscar_apenas_sem_gpt(payload: AskRequest):
    try:
        question = payload.question.strip()
        vector = gerar_embedding_openai(question)

        results = client.collections.get("Article").query.near_vector(
            near_vector=vector, limit=5
        )

        articles = results.objects or []
        artigos_resumidos = [
            {
                "title": a.properties.get("title", ""),
                "url": a.properties.get("url", "#"),
                "snippet": a.properties.get("content", "")[:300] + "...",
            }
            for a in articles
        ]
        return {"resposta": None, "artigos": artigos_resumidos}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# 📨 Endpoint atualizado para buscar tickets resolvidos/fechados do Movidesk
@app.get("/movidesk-tickets")
async def buscar_tickets_movidesk(limite: int = 10):
    try:
        token = os.getenv("MOVI_TOKEN")
        base_url = os.getenv("MOVI_TICKET_URL")
        if not token or not base_url:
            raise ValueError("MOVI_TOKEN ou MOVI_TICKET_URL não configurados no .env")

        headers = {"Content-Type": "application/json"}
        params = {
            "token": token,
            "$select": "id,subject,category,createdDate,status,actions",
            "$expand": "actions",
            "$orderby": "createdDate desc",
            "$top": 50,  # busca mais para filtrar depois
        }

        response = requests.get(base_url, headers=headers, params=params)
        if (response.status_code != 200):
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "erro": "Erro ao buscar tickets",
                    "status_code": response.status_code,
                    "detalhes": response.text,
                },
            )

        todos_tickets = response.json()

        # 🎯 Filtro com base no campo "status"
        status_validos = {"Resolvido", "Fechado"}
        filtrados = [
            t for t in todos_tickets if t.get("status", "").strip() in status_validos
        ]

        return {"tickets": filtrados[:limite]}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"erro": "Erro inesperado no servidor", "detalhes": str(e)},
        )


@app.post("/weaviate-artigos")
async def inserir_artigo_weaviate(artigo: ArticleInput):
    try:
        collection = get_weaviate_client.collections.get("Article")
        collection.data.insert(
            properties={
                "title": artigo.title,
                "content": artigo.content,
                "question": artigo.question,
                "url": artigo.source,
                "type": artigo.type,
            }
        )
        return {"status": "sucesso"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weaviate-artigos")
async def listar_artigos_weaviate():
    try:
        collection = get_weaviate_client.collections.get("Article")
        results = collection.query.fetch_objects(limit=50)

        artigos = []
        for obj in results.objects:
            props = obj.properties
            artigos.append(
                {
                    "title": props.get("title", "Sem título"),
                    "type": props.get("type", ""),
                    "question": props.get("question", ""),
                }
            )

        return artigos
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="Erro ao buscar artigos no Weaviate."
        )


# Buscar Prompt


@app.get("/prompt")
async def get_prompt(nome: str = Query("padrao")):
    try:
        result = (
            supabase.table("prompts")
            .select("conteudo")
            .eq("nome", nome)
            .eq("ativo", True)
            .limit(1)
            .execute()
        )

        if result.data:
            return {"prompt": result.data[0]["conteudo"]}
        else:
            raise HTTPException(
                status_code=404, detail=f"Prompt '{nome}' não encontrado."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Centralizar a função de construção do prompt dinâmico
def construir_prompt_base(pergunta: str, context: str = "", historico: str = "") -> str:
    prompt_base = buscar_prompt()
    return (
        prompt_base.replace("{question}", pergunta)
        .replace("{context}", context)
        .replace("{historico_texto}", historico)
    )

@app.post("/prompt")
async def salvar_prompt(payload: PromptAtualizacao):
    nome = payload.nome.strip()
    conteudo = payload.conteudo.strip()

    try:
        # Desativa prompts antigos com o mesmo nome
        existentes = supabase.table("prompts").select("id").eq("nome", nome).execute()
        for existente in existentes.data:
            prompt_id = existente["id"]
            supabase.table("prompts").update({"ativo": False}).eq("id", prompt_id).execute()

        # Insere novo
        novo = {"nome": nome, "conteudo": conteudo, "ativo": True}
        supabase.table("prompts").insert(novo).execute()

        return {"message": f"Prompt '{nome}' salvo com sucesso ✅"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Função que retorna a sessão ativa ou cria uma nova
def obter_sessao_ativa(usuario_id: str) -> str:
    try:
        resultado = (
            supabase.table("mensagens")
            .select("sessao_id, created_at")
            .eq("usuario_id", usuario_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if resultado.data:
            ultima = resultado.data[0]
            data_ultima = datetime.fromisoformat(ultima["created_at"])
            if datetime.utcnow() - data_ultima < timedelta(minutes=10):
                return ultima["sessao_id"]

        return f"{usuario_id}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    except Exception as e:
        print(f"Erro ao obter sessão ativa: {e}")
        return f"{usuario_id}-{uuid.uuid4()}"


# Função que salva a mensagem com a sessão


def salvar_mensagem(usuario_id, pergunta, resposta, embedding, sessao_id=None):
    try:
        if not sessao_id:
            raise ValueError("Sessão não definida para salvar a mensagem.")

        # Gerar embedding se não foi fornecido
        if embedding is None and pergunta:
            embedding = gerar_embedding_openai(pergunta)

        payload = {
            "usuario_id": usuario_id,
            "sessao_id": sessao_id,
            "pergunta": pergunta,
            "resposta": resposta,
            "embedding": embedding,
            "created_at": get_brazil_time().isoformat(),
        }

        print(f"💾 Salvando mensagem: {payload}")
        supabase = get_supabase_client()
        supabase.table("mensagens").insert(payload).execute()
        print("✅ Mensagem salva com sucesso.")

    except Exception as e:
        print(f"❌ Erro ao salvar mensagem: {e}")
        raise HTTPException(status_code=500, detail="Erro ao salvar mensagem.")


# Novo endpoint para listar conversas por sessão
from fastapi import Query


@app.get("/conversas")
def listar_conversas(usuario_id: str = Query(...)):
    try:
        result = (
            supabase.table("mensagens")
            .select("sessao_id, pergunta, resposta, data")
            .eq("usuario_id", usuario_id)
            .order("data", desc=True)
            .execute()
        )

        conversas = {}
        for msg in result.data:
            sid = msg["sessao_id"]
            if sid not in conversas:
                conversas[sid] = []
            conversas[sid].append(
                {
                    "pergunta": msg["pergunta"],
                    "resposta": msg["resposta"],
                    "data": msg["data"],
                }
            )

        # Converte em lista e ordena por data da primeira mensagem da sessão
        lista_sessoes = [
            {"sessao_id": sid, "mensagens": msgs, "data_inicio": msgs[-1]["data"]}
            for sid, msgs in conversas.items()
        ]
        lista_sessoes.sort(key=lambda x: x["data_inicio"], reverse=True)

        return lista_sessoes

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessoes")
def listar_sessoes(usuario_id: str = Query(...)):
    try:
        supabase = get_supabase_client()

        sessoes = (
            supabase.table("sessoes")
            .select("id, inicio, fim")
            .eq("usuario_id", usuario_id)
            .order("inicio", desc=True)
            .execute()
        ).data or []

        conversas_por_sessao = (
            supabase.table("mensagens")
            .select("sessao_id, pergunta, resposta, created_at")
            .eq("usuario_id", usuario_id)
            .order("created_at", asc=True)
            .execute()
        ).data or []

        sessao_dict = {}
        for msg in conversas_por_sessao:
            sid = msg["sessao_id"]
            if sid not in sessao_dict:
                sessao_dict[sid] = []
            sessao_dict[sid].append({
                "pergunta": msg["pergunta"],
                "resposta": msg["resposta"],
                "created_at": msg["created_at"]
            })

        resultado = []
        for s in sessoes:
            resultado.append({
                "sessao_id": s["id"],
                "inicio": s["inicio"],
                "fim": s.get("fim"),
                "mensagens": sessao_dict.get(s["id"], [])
            })

        return {"sessoes": resultado}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def encerrar_sessao_antiga(supabase, sessao_id):
    try:
        supabase.table("sessoes").update({
            "fim": datetime.utcnow().isoformat()
        }).eq("id", sessao_id).execute()
    except Exception as e:
        print(f"Erro ao encerrar sessão {sessao_id}: {e}")


def obter_ou_criar_sessao(supabase, usuario_id: str) -> int:
    try:
        print(f"🔍 Verificando sessão ativa para o usuário: {usuario_id}")
        resultado = (
            supabase.table("sessoes")
            .select("id, inicio")
            .eq("usuario_id", usuario_id)
            .order("inicio", desc=True)
            .limit(1)
            .execute()
        )

        if resultado.data:
            ultima_sessao = resultado.data[0]
            print(f"✅ Sessão encontrada: {ultima_sessao}")
            data_inicio = datetime.fromisoformat(ultima_sessao["inicio"])
            # Comparação usando fuso horário correto
            if get_brazil_time() - data_inicio < timedelta(minutes=10):
                print(f"🔄 Sessão ainda ativa: {ultima_sessao['id']}")
                return ultima_sessao["id"]

            print(f"⏳ Encerrando sessão antiga: {ultima_sessao['id']}")
            encerrar_sessao_antiga(supabase, ultima_sessao["id"])

        # Criar nova sessão com ID auto-incrementado
        nova_sessao = supabase.table("sessoes").insert({
            "usuario_id": usuario_id,
            "inicio": get_brazil_time().isoformat()
        }).execute()
        
        return nova_sessao.data[0]["id"]

    except Exception as e:
        print(f"❌ Erro ao obter ou criar sessão: {e}")
        raise HTTPException(status_code=500, detail="Erro ao gerenciar sessão.")


def get_brazil_time():
    """Retorna datetime atual no fuso horário do Brasil (UTC-3)"""
    return datetime.utcnow() - timedelta(hours=3)