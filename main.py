from dotenv import load_dotenv
load_dotenv()

# 📦 Imports padrão
import os
import json
import time
import traceback
import requests
from datetime import datetime
from collections import Counter
from contextlib import asynccontextmanager

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
from dateutil.parser import isoparse  # coloque no topo se ainda não estiver importado

# 🧩 Integrações
from supabase_utils import salvar_mensagem, busca_mensagens_por_embedding
from supabase import create_client



# 🔐 Variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ✅ Criação única do client Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

USE_OPENAI = True

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
    usuario_id: Optional[str] = "amigo"
    usuario_nome: Optional[str] = None  # 🔹 Novo campo
    historico: Optional[List[dict]] = []


class Curadoria(BaseModel):
    ticket_id: int
    curador: str
    question: str
    answer: str

class TextoLivre(BaseModel):
    texto: str
    instrucoes: Optional[str] = None

# 🔁 Ciclo de vida da aplicação
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n✅ Conectando ao Weaviate...")
    global client   
    client = weaviate.connect_to_wcs(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        skip_init_checks=True
    )
    print("🔌 Conectado ao Weaviate.")

    if USE_OPENAI:
        global openai
        openai = OpenAI(api_key=OPENAI_API_KEY)
        print("🔐 OpenAI conectado.")

    yield

    print("🛑 Fechando conexão com o Weaviate...")
    client.close()

app = FastAPI(lifespan=lifespan)


def carregar_parametros_ia():
    dados = supabase.table("parametros").select("nome", "valor").execute().data
    return {item["nome"]: item["valor"] for item in dados}

def salvar_parametros_ia(modelo, temperatura, top_p):
    supabase.table("parametros").upsert([
        {"nome": "modelo", "valor": modelo},
        {"nome": "temperatura", "valor": str(temperatura)},
        {"nome": "top_p", "valor": str(top_p)}
    ]).execute()

def carregar_social_keywords():
    try:
        response = supabase.table("parametros").select("valor").eq("nome", "social_keywords").execute()
        if response.data and response.data[0].get("valor"):
            keywords_csv = response.data[0]["valor"]
            palavras = [k.strip().lower() for k in keywords_csv.split(",") if k.strip()]
            if palavras:
                return palavras
    except Exception as e:
        print(f"Erro ao buscar social_keywords: {e}")  # 👈 log visível

    # fallback padrão
    return [
        "oi", "olá", "bom dia", "boa tarde", "boa noite", "tudo bem",
        "como vai", "e aí", "quem é você", "seu nome", "você é real",
        "você é humano", "me conta sobre você", "como você está"
    ]

def eh_pergunta_social(pergunta: str) -> bool:
    pergunta = pergunta.lower()
    palavras = carregar_social_keywords()
    return any(p in pergunta for p in palavras)



def responder_direto_sem_rag(pergunta: str, user_id: str = "amigo") -> str:
    nome = user_id or "amigo"
    prompt = f"""
Você é um assistente virtual acolhedor da Sisand. O nome do usuário é {nome}.
Responda com empatia e simpatia. Não inclua informações técnicas ou da base de conhecimento.
Pergunta: {pergunta}
"""
    resposta = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resposta.choices[0].message.content.strip()


def gerar_resposta_social(pergunta: str, nome_usuario: str = "amigo") -> str:
    pergunta_lower = pergunta.lower()

    if "tudo bem" in pergunta_lower:
        return f"Oi, {nome_usuario}! Aqui é o Sisandinho 😊 Tudo ótimo por aqui e com você? Me conta como posso te ajudar!"
    elif "boa tarde" in pergunta_lower:
        return f"Boa tarde, {nome_usuario}! ☀️ Eu sou o Sisandinho e tô aqui pra te dar uma força no que precisar!"
    elif "bom dia" in pergunta_lower:
        return f"Bom dia, {nome_usuario}! 🌞 Aqui é o Sisandinho, pronto pra deixar seu dia mais fácil! Como posso ajudar?"
    elif "boa noite" in pergunta_lower:
        return f"Boa noite, {nome_usuario}! 🌙 Sou o Sisandinho, seu ajudante digital. Vamos resolver o que for preciso!"
    elif "oi" in pergunta_lower or "olá" in pergunta_lower:
        return f"Olá, {nome_usuario}! 👋 Eu sou o Sisandinho, seu agente de suporte da Sisand. Como posso ajudar você hoje?"
    elif "como vai" in pergunta_lower or "e aí" in pergunta_lower:
        return f"E aí, {nome_usuario}! 😄 Eu tô aqui, firme e forte, preparado pra te ajudar no que for!"
    else:
        return f"Oi, {nome_usuario}! Aqui é o Sisandinho. Pronto pra ajudar você com o que precisar! 💙"



CURADORIA_FILE = "curadorias.json"

# Endpoint para retornar embeddings dos artigos
@app.get("/embeddings")
async def listar_embeddings(limit: int = 100):
    collection = client.collections.get("Article")
    artigos = collection.query.fetch_objects(limit=limit, include_vector=True)
    resposta = []
    for art in artigos.objects:
        vetor = art.vector
        if isinstance(vetor, dict):
            vetor = vetor.get("default", [])
        if not vetor or not isinstance(vetor, list):
            continue
        resposta.append({
            "title": art.properties.get("title", "Sem Título"),
            "url": art.properties.get("url", "#"),
            "vector": vetor
        })
    return resposta



@app.post("/gpt-curadoria")
async def gerar_curadoria_com_gpt(payload: TextoLivre, nome: str = Query(default="curadoria")):
    try:
        texto = payload.texto.strip()
        if not texto:
            raise ValueError("Texto da curadoria está vazio.")

        # Busca do prompt dinâmico na Supabase
        prompt_base = buscar_prompt(nome=nome)
        if "Prompt não encontrado" in prompt_base:
            raise ValueError("Prompt 'curadoria' não encontrado no Supabase.")

        prompt = prompt_base \
            .replace("{texto}", texto) \
            .replace("{question}", "") \
            .replace("{context}", "") \
            .replace("{historico_texto}", "")

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você atua como especialista em curadoria de atendimentos técnicos para sistemas ERP."},
                {"role": "user", "content": prompt.strip()}
            ]
        )

        texto_gerado = completion.choices[0].message.content.strip()
        dados = json.loads(texto_gerado)
        return dados

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao interpretar JSON gerado pela IA.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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

        prompt = f"""
Você é um especialista em atendimento técnico. Abaixo está um resumo informal de uma situação de atendimento:

\"\"\"{texto}\"\"\"

A partir disso, gere uma estrutura com os seguintes campos:
- Pergunta do cliente
- Solução aplicada
- Diagnóstico técnico (se aplicável)
- Resultado final (benefício ou conclusão)

Responda apenas com um JSON no formato:
{{
  "pergunta": "...",
  "solucao": "...",
  "diagnostico": "...",
  "resultado": "..."
}}
        """

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um especialista em atendimento técnico e curadoria de conhecimento."},
                {"role": "user", "content": prompt}
            ]
        )

        texto_gerado = completion.choices[0].message.content.strip()
        import json
        dados = json.loads(texto_gerado)
        return dados

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/curadoria")
def registrar_curadoria(curadoria: Curadoria):
    curadoria_data = {
        "ticket_id": curadoria.ticket_id,
        "curador": curadoria.curador,
        "question": curadoria.question,
        "answer": curadoria.answer,
        "data": datetime.now().isoformat()
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
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding





# --- FASTAPI: FUNÇÃO PARA BUSCAR PROMPT ---

def buscar_prompt(nome: str = "padrao") -> str:
    try:
        result = supabase.table("prompts") \
            .select("conteudo") \
            .eq("nome", nome) \
            .eq("ativo", True) \
            .limit(1).execute()

        if result.data:
            return result.data[0]["conteudo"]
    except Exception as e:
        print("Erro ao buscar prompt:", e)

    return "Prompt não encontrado."



def buscar_artigos_semanticamente(pergunta, personalidade="", usar_gpt=True, limite=3):
    try:
        vector = gerar_embedding_openai(pergunta)

        results = client.collections.get("Article").query.near_vector(
            near_vector=vector,
            limit=limite
        )
        artigos = results.objects

        if not artigos:
            return {"resposta": "Nenhum artigo relevante encontrado.", "artigos": []}

        context = "\n\n".join([
            f"{a.properties['title']}:\n{a.properties['content'][:1000]}" for a in artigos
        ])
        artigos_resumidos = [
            {
                "title": a.properties["title"],
                "url": a.properties.get("url", "#"),
                "snippet": a.properties["content"][:300] + "..."
            }
            for a in artigos
        ]

        artigos_titulos = [a.properties["title"] for a in artigos]
        artigos_utilizados.update(artigos_titulos)

        if usar_gpt and USE_OPENAI:
            import time
            inicio = time.time()

            prompt = f"""
{personalidade}

Abaixo estão alguns artigos técnicos sobre o sistema da Sisand:
{context}

Com base nesses artigos, responda de forma clara e útil à seguinte pergunta:

{pergunta}

Resposta:
"""
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo" if len(context) > 3000 else "gpt-4",
                messages=[
                    {"role": "system", "content": "Você é um assistente técnico especialista em sistemas ERP."},
                    {"role": "user", "content": prompt.strip()}
                ]
            )
            fim = time.time()
            tempo_resposta = round(fim - inicio, 2)
            tempos_resposta.append(tempo_resposta)
            perguntas_registradas.append(pergunta)

            return {
                "resposta": completion.choices[0].message.content.strip(),
                "artigos": artigos_resumidos,
                "tempo": tempo_resposta
            }

        return {
            "resposta": context,
            "artigos": artigos_resumidos
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-only")
async def buscar_apenas_sem_gpt(payload: AskRequest):
    try:
        question = payload.question.strip()
        vector = gerar_embedding_openai(question)

        results = client.collections.get("Article").query.near_vector(
            near_vector=vector,
            limit=5
        )

        articles = results.objects or []
        artigos_resumidos = [
            {
                "title": a.properties.get("title", ""),
                "url": a.properties.get("url", "#"),
                "snippet": a.properties.get("content", "")[:300] + "..."
            }
            for a in articles
        ]
        return {"resposta": None, "artigos": artigos_resumidos}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ask")
def ask_question(payload: AskRequest):
    try:
        question = payload.question.strip()
        personalidade = payload.personalidade.strip()
        use_gpt = payload.use_gpt
        limit = payload.limit
        modelo = getattr(payload, 'modelo', None)
        usuario_id = getattr(payload, "usuario_id", "anonimo")
        nome_usuario = payload.usuario_nome or payload.usuario_id or "amigo"

        # ===== Checagem para pergunta social =====
        if eh_pergunta_social(question):
            print("✅ Pergunta social detectada. Respondendo sem embeddings ou busca.")
            resposta_social = gerar_resposta_social(question, nome_usuario)
            if not resposta_social:
                resposta_social = responder_direto_sem_rag(question, user_id=usuario_id)

            # 💾 Salva como sessão também
            sessao_id = obter_sessao_ativa(usuario_id)
            salvar_mensagem(usuario_id, question, resposta_social, None, sessao_id=sessao_id)

            return {
                "resposta": resposta_social,
                "artigos": [],
                "tempo": 0.1,
                "sisandinho": True
            }

        # ===== Fluxo RAG (GPT com contexto) =====
        inicio_embedding = time.time()
        vector = gerar_embedding_openai(question)
        tempo_embedding = time.time() - inicio_embedding

        mensagens_supabase = busca_mensagens_por_embedding(vector, min_similarity=0.75)
        mensagens = mensagens_supabase.get("dados", [])

        inicio_busca = time.time()
        results = client.collections.get("Article").query.near_vector(
            near_vector=vector,
            limit=limit,
            return_properties=["title", "url", "content"],
            include_vector=True
        )
        tempo_busca = time.time() - inicio_busca

        articles = results.objects or []
        artigos_filtrados, vetores_artigos, context_pieces = [], [], []

        for a in articles:
            vetor = a.vector.get('default') if hasattr(a, 'vector') and isinstance(a.vector, dict) else a.vector
            if isinstance(vetor, list) and all(isinstance(x, (int, float)) for x in vetor):
                artigos_filtrados.append(a)
                vetores_artigos.append(vetor)

        if not artigos_filtrados:
            return {
                "resposta": None,
                "artigos": [],
                "tempo": round(tempo_embedding + tempo_busca, 2)
            }

        sim_scores = cosine_similarity([vector], vetores_artigos)[0]
        artigos_resumidos = []
        total_chars = 0

        for artigo, score in zip(artigos_filtrados, sim_scores):
            title = artigo.properties.get("title", "")
            url = artigo.properties.get("url", "#")
            content = artigo.properties.get("content", "").replace("\n", " ").strip()
            artigos_resumidos.append({
                "title": title,
                "url": url,
                "similaridade": round(score * 100, 2)
            })
            if total_chars < 24000:
                trecho = f"Título: {title}\nConteúdo:\n{content[:3000]}"
                context_pieces.append(trecho)
                total_chars += len(trecho)

        historico = payload.historico[-3:]
        historico_texto = "\n\n".join([
            f"Usuário: {h['pergunta']}\nAgente: {h['resposta']}"
            for h in historico if 'pergunta' in h and 'resposta' in h
        ])

        modelo_usado = modelo if modelo in ["gpt-3.5-turbo", "gpt-4"] else "gpt-3.5-turbo"
        prompt_base = buscar_prompt("padrao")
        context = "\n\n".join(context_pieces)
        prompt = prompt_base.replace("{question}", question).replace("{context}", context).replace("{historico_texto}", historico_texto)

        # 🧠 Busca (ou cria) uma sessão ativa com base no tempo da última interação
        sessao_id = obter_sessao_ativa(usuario_id)

        # ⏱️ Marca o tempo antes de chamar o GPT
        inicio_gpt = time.time()
        completion = openai.chat.completions.create(
            model=modelo_usado,
            messages=[
                {"role": "system", "content": "Você é um assistente técnico especialista em sistemas ERP."},
                {"role": "user", "content": prompt.strip()}
            ]
        )
        tempo_gpt = time.time() - inicio_gpt

        # ✏️ Processa a resposta
        resposta_final = completion.choices[0].message.content.strip()

        # 💾 Salva no Supabase, agora com a sessão atual
        salvar_mensagem(usuario_id, question, resposta_final, vector, sessao_id=sessao_id)

        return {
            "resposta": resposta_final,
            "artigos": artigos_resumidos,
            "tempo": round(tempo_embedding + tempo_busca + tempo_gpt, 2),
            "sisandinho": False
        }

    except Exception as e:
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

class FeedbackAberto(BaseModel):
    pergunta: str
    resposta: str
    comentario: str

@app.post("/feedback")
async def salvar_feedback(feedback: FeedbackAberto):
    feedbacks_abertos.append(feedback.dict())
    return {"status": "comentário salvo"}

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
            "$top": 50  # busca mais para filtrar depois
        }

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            return JSONResponse(status_code=response.status_code, content={
                "erro": "Erro ao buscar tickets",
                "status_code": response.status_code,
                "detalhes": response.text
            })

        todos_tickets = response.json()

        # 🎯 Filtro com base no campo "status"
        status_validos = {"Resolvido", "Fechado"}
        filtrados = [
            t for t in todos_tickets
            if t.get("status", "").strip() in status_validos
        ]

        return {"tickets": filtrados[:limite]}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "erro": "Erro inesperado no servidor",
            "detalhes": str(e)
        })


class ArticleInput(BaseModel):
    title: str
    content: str
    question: str = ""
    source: str = ""
    type: str = "resposta_ticket"

@app.post("/weaviate-save")
async def salvar_no_weaviate(artigo: ArticleInput):
    try:
        collection = client.collections.get("Article")
        collection.data.insert(properties={
            "title": artigo.title,
            "content": artigo.content,
            "question": artigo.question,
            "url": artigo.source,
            "type": artigo.type
        })
        return {"status": "sucesso"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/weaviate-artigos")
async def listar_artigos_salvos():
    try:
        collection = client.collections.get("Article")
        results = collection.query.fetch_objects(limit=50)

        artigos = []
        for obj in results.objects:
            props = obj.properties
            artigos.append({
                "title": props.get("title", "Sem título"),
                "type": props.get("type", ""),
                "question": props.get("question", "")
            })

        return artigos
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro ao buscar artigos no Weaviate.")

# Buscar Prompt


@app.get("/prompt")
async def get_prompt(nome: str = Query("padrao")):
    try:
        result = supabase.table("prompts") \
            .select("conteudo") \
            .eq("nome", nome) \
            .eq("ativo", True) \
            .limit(1).execute()

        if result.data:
            return {"prompt": result.data[0]["conteudo"]}
        else:
            raise HTTPException(status_code=404, detail=f"Prompt '{nome}' não encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#Centralizar a função que monta o prompt dinâmico
def construir_prompt_base(pergunta: str, context: str, historico: str = "") -> str:
    prompt_base = buscar_prompt("padrao")

    return prompt_base \
        .replace("{question}", pergunta.strip()) \
        .replace("{context}", context.strip()) \
        .replace("{historico_texto}", historico.strip())




# Atualiza o prompt


class PromptAtualizacao(BaseModel):
    nome: str
    novo_prompt: str

@app.post("/prompt")
async def salvar_prompt(payload: PromptAtualizacao):
    nome = payload.nome.strip()
    conteudo = payload.novo_prompt.strip()

    try:
        # Verifica se já existe ativo
        existentes = supabase.table("prompts") \
            .select("id") \
            .eq("nome", nome) \
            .eq("ativo", True).execute()

        if existentes.data:
            # Desativa o anterior
            prompt_id = existentes.data[0]["id"]
            supabase.table("prompts").update({"ativo": False}).eq("id", prompt_id).execute()

        # Insere novo
        novo = {
            "nome": nome,
            "conteudo": conteudo,
            "ativo": True
        }
        supabase.table("prompts").insert(novo).execute()
        return {"mensagem": f"Prompt '{nome}' atualizado com sucesso ✅"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Função que retorna a sessão ativa (mesmo ID se for dentro de 10min)
from datetime import datetime, timedelta
import uuid

def obter_sessao_ativa(usuario_id: str) -> str:
    try:
        resultado = supabase.table("mensagens") \
            .select("sessao_id, created_at") \
            .eq("usuario_id", usuario_id) \
            .order("created_at", desc=True) \
            .limit(1).execute()

        if resultado.data:
            ultima = resultado.data[0]
            data_ultima = datetime.fromisoformat(ultima["created_at"].replace("Z", ""))
            if datetime.utcnow() - data_ultima < timedelta(minutes=10):
                return ultima["sessao_id"]

        return f"{usuario_id}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    except Exception as e:
        print(f"Erro ao obter sessão ativa: {e}")
        return f"{usuario_id}-{uuid.uuid4()}"


# Função que salva a mensagem com a sessão

def salvar_mensagem(usuario_id, pergunta, resposta, embedding, sessao_id=None):
    if not sessao_id:
        sessao_id = obter_sessao_ativa(usuario_id)

    payload = {
        "usuario_id": usuario_id,
        "sessao_id": sessao_id,
        "pergunta": pergunta,
        "resposta": resposta,
        "embedding": embedding,
        "created_at": datetime.utcnow().isoformat()
    }

    try:
        supabase.table("mensagens").insert(payload).execute()
    except Exception as e:
        print(f"Erro ao salvar mensagem: {e}")


# Novo endpoint para listar conversas por sessão
from fastapi import Query

@app.get("/conversas")
def listar_conversas(usuario_id: str = Query(...)):
    try:
        result = supabase.table("mensagens") \
            .select("sessao_id", "pergunta", "resposta", "data") \
            .eq("usuario_id", usuario_id) \
            .order("data", desc=True).execute()

        conversas = {}
        for msg in result.data:
            sid = msg["sessao_id"]
            if sid not in conversas:
                conversas[sid] = []
            conversas[sid].append({
                "pergunta": msg["pergunta"],
                "resposta": msg["resposta"],
                "data": msg["data"]
            })

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
        mensagens = (
            supabase.table("mensagens")
            .select("sessao_id, pergunta, resposta, created_at")
            .eq("usuario_id", usuario_id)
            .order("created_at")  # ✅ Removido asc=True
            .execute()
        ).data or []

        sessoes_dict = {}
        for msg in mensagens:
            sessao_id = msg["sessao_id"]
            if sessao_id not in sessoes_dict:
                sessoes_dict[sessao_id] = []
            sessoes_dict[sessao_id].append({
                "pergunta": msg["pergunta"],
                "resposta": msg["resposta"],
                "created_at": msg["created_at"]
            })

        sessoes = [
            {
                "sessao_id": sessao_id,
                "inicio": msgs[0]["created_at"],
                "mensagens": msgs
            }
            for sessao_id, msgs in sessoes_dict.items()
        ]

        sessoes.sort(key=lambda x: x["inicio"], reverse=True)

        return {"sessoes": sessoes}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


