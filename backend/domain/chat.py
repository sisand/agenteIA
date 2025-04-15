from fastapi import APIRouter, HTTPException
from backend.clients import get_weaviate_client, get_openai_client, gerar_embedding_openai, get_supabase_client
from backend.domain.models import Pergunta, Resposta
import logging
import time
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)

router = APIRouter()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def buscar_artigos_relacionados(pergunta: str):
    """
    Busca artigos relacionados no Weaviate usando embeddings.
    """
    try:
        client = get_weaviate_client()
        
        # Gerar embedding para a pergunta
        query_vector = gerar_embedding_openai(pergunta)
        
        # Buscar artigos relacionados usando near_vector
        results = client.collections.get("Article").query.near_vector(
            near_vector=query_vector,
            limit=3,
            return_properties=["title", "url"]
        )
        
        artigos = []
        for artigo in results.objects:
            artigos.append({
                "title": artigo.properties.get("title", "Sem Título"),
                "url": artigo.properties.get("url", "#")
            })
        return artigos
    except Exception as e:
        logging.error(f"Erro ao buscar artigos relacionados: {str(e)}")
        return []

async def buscar_prompt_sistema(personalidade: str = ""):
    """
    Busca o prompt do sistema baseado na personalidade.
    """
    try:
        supabase = get_supabase_client()
        nome_prompt = "padrao"  # Sempre usar o prompt padrão
        
        result = (
            supabase.table("prompts")
            .select("conteudo")
            .eq("nome", nome_prompt)
            .eq("ativo", True)
            .limit(1)
            .execute()
        )
        
        if result.data:
            return result.data[0]["conteudo"]
        return "Você é um assistente especializado em Sisand."
    except Exception as e:
        logging.error(f"Erro ao buscar prompt do sistema: {str(e)}")
        return "Você é um assistente especializado em Sisand."

def gerar_resposta_ia(pergunta: str):
    """
    Gera uma resposta usando OpenAI com o prompt configurado.
    """
    try:
        # Carregar configurações
        supabase = get_supabase_client()
        config = (
            supabase.table("parametros")
            .select("valor")
            .eq("nome", "prompt_chat_padrao")
            .limit(1)
            .execute()
        )
        
        prompt_padrao = config.data[0]["valor"] if config.data else "padrao"
        
        # Carregar prompt
        prompt_result = (
            supabase.table("prompts")
            .select("conteudo")
            .eq("nome", prompt_padrao)
            .eq("ativo", True)
            .limit(1)
            .execute()
        )
        
        prompt_conteudo = prompt_result.data[0]["conteudo"] if prompt_result.data else ""
        
        # Gerar resposta
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_conteudo},
                {"role": "user", "content": pergunta},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erro ao gerar resposta com OpenAI: {str(e)}")
        return "Desculpe, não consegui gerar uma resposta no momento."

@router.post("/ask", response_model=Resposta)
async def processar_pergunta(payload: Pergunta):
    """
    Processa a pergunta enviada pelo usuário e retorna uma resposta baseada em IA e artigos relacionados.
    """
    try:
        logging.info(f"Payload recebido: {payload.model_dump()}")
        inicio = time.time()

        # Busca artigos relacionados
        artigos_relacionados = buscar_artigos_relacionados(payload.question)
        
        # Formatar histórico se existir
        historico_texto = ""
        if payload.historico:
            historico_texto = "\n".join([
                f"Usuário: {msg.get('pergunta', '')}\nAssistente: {msg.get('resposta', '')}"
                for msg in payload.historico
            ])

        # Preparar contexto com os artigos encontrados
        contexto_artigos = "\n\n".join([
            f"Artigo: {artigo['title']}\nURL: {artigo['url']}"
            for artigo in artigos_relacionados
        ]) if artigos_relacionados else "Nenhum artigo relacionado encontrado."

        # Garantir que o prompt base seja carregado
        prompt_base = await buscar_prompt_sistema()
        
        # Montar prompt completo com substituição de variáveis
        prompt_completo = (
            prompt_base
            .replace("{historico_texto}", historico_texto)
            .replace("{context}", contexto_artigos)
            .replace("{question}", payload.question)
        )

        # Log do prompt para debug
        logging.info(f"Prompt completo montado: {prompt_completo}")

        # Gera resposta usando IA
        client = get_openai_client()
        response = client.chat.completions.create(
            model=payload.modelo,
            messages=[
                {"role": "system", "content": prompt_completo},
                {"role": "user", "content": payload.question},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        
        resposta_ia = response.choices[0].message.content.strip()
        tempo_resposta = time.time() - inicio

        # Retornar resposta incluindo o prompt usado
        return Resposta(
            resposta=resposta_ia, 
            artigos=artigos_relacionados, 
            tempo=tempo_resposta,
            prompt_usado=prompt_completo  # Garantir que o prompt completo seja retornado
        )

    except Exception as e:
        logging.error(f"Erro ao processar a pergunta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")
