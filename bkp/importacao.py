import logging
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import time
from fastapi import APIRouter, HTTPException
from backend.clients import get_weaviate_client, get_openai_client
from backend.domain.artigos import MovideskArticle, sanitize_content, format_movidesk_url

router = APIRouter()
logger = logging.getLogger(__name__)

def from_iso_to_date(date_str: str) -> Optional[str]:
    """Converte data ISO para formato aceito pelo Weaviate"""
    if not date_str:
        return None
    ds = date_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ds)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None

async def buscar_lista_artigos(pagina: int = 1, limite: int = 50) -> List[Dict[str, Any]]:
    """Busca lista paginada de artigos do Movidesk."""
    try:
        token = os.getenv("MOVI_TOKEN")
        base_url = os.getenv("MOVI_LIST_URL")
        
        params = {
            "token": token,
            "$top": limite,
            "$skip": (pagina - 1) * limite,
            "$orderby": "id desc"
        }
        
        response = requests.get(
            base_url,
            params=params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            return []
            
        return items

    except Exception as e:
        logger.error(f"❌ Erro ao buscar lista de artigos: {str(e)}")
        return []

async def buscar_todos_artigos(limite_por_pagina: int = 50) -> List[Dict[str, Any]]:
    """Busca todos os artigos do Movidesk, paginando até o final."""
    todos_artigos = []
    pagina = 1

    while True:
        artigos = await buscar_lista_artigos(pagina, limite_por_pagina)
        if not artigos:  # Parar se não houver mais artigos
            logger.info(f"📋 Nenhum artigo encontrado na página {pagina}. Finalizando paginação.")
            break

        todos_artigos.extend(artigos)
        logger.info(f"📄 Página {pagina}: {len(artigos)} artigos encontrados")

        # Parar se o número de artigos retornados for menor que o limite por página
        if len(artigos) < limite_por_pagina:
            logger.info(f"📋 Última página alcançada ({pagina}). Finalizando paginação.")
            break

        pagina += 1

    logger.info(f"📋 Total de artigos encontrados: {len(todos_artigos)}")
    return todos_artigos

async def buscar_detalhes_artigo(artigo_id: int) -> Optional[Dict[str, Any]]:
    """Busca detalhes de um artigo específico."""
    try:
        token = os.getenv("MOVI_TOKEN")
        base_url = os.getenv("MOVI_DETAIL_URL", "https://api.movidesk.com/public/v1/article")
        
        if not token or not base_url:
            raise ValueError("MOVI_TOKEN ou MOVI_DETAIL_URL não configurados")

        url = f"{base_url}/{artigo_id}"
        headers = {"Content-Type": "application/json"}
        params = {"token": token}
        
        logger.info(f"🔍 Buscando detalhes do artigo ID {artigo_id} em: {url}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        # Garantir que a resposta seja um JSON válido
        artigo = response.json()
        if not isinstance(artigo, dict):
            logger.error(f"❌ Resposta inesperada ao buscar detalhes do artigo {artigo_id}: {artigo}")
            return None

        logger.info(f"✅ Detalhes do artigo {artigo_id} obtidos com sucesso")
        return artigo

    except Exception as e:
        logger.error(f"❌ Erro ao buscar detalhes do artigo {artigo_id}: {str(e)}")
        return None

async def gerar_embedding_conteudo(content: str) -> Optional[List[float]]:
    """Gera embeddings para o conteúdo do artigo usando OpenAI."""
    try:
        openai_client = get_openai_client()
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=content
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"❌ Erro ao gerar embedding: {str(e)}")
        return None

async def verificar_artigo_existe(collection, id_movidesk: int) -> Optional[str]:
    """
    Verifica se um artigo já existe no Weaviate pelo id_movidesk.
    """
    try:
        where_filter = {
            "path": ["movidesk_id"],
            "operator": "Equal",
            "valueInt": id_movidesk
        }
        
        result = (
            collection.query
            .get("movidesk_id")
            .with_where(where_filter)
            .do()
        )

        if result and result['data']['Get']['Article']:
            return result['data']['Get']['Article'][0]['movidesk_id']
        return None
        
    except Exception as e:
        logger.error(f"❌ Erro ao verificar existência do artigo: {str(e)}")
        return None

async def verificar_e_criar_schema(client, reset_base: bool = True):
    """Verifica e cria o schema se necessário, com opção de resetar a base."""
    try:
        # Verificar se collection existe
        collection_exists = client.collections.exists("Article")
        
        # Resetar a base se necessário
        if reset_base and collection_exists:
            try:
                client.collections.delete("Article")
                logger.info("🗑️ Collection 'Article' excluída com sucesso")
                await asyncio.sleep(2)  # Aguardar a exclusão ser processada
                collection_exists = False
            except Exception as e:
                logger.error(f"❌ Erro ao excluir collection: {str(e)}")
                raise

        # Criar collection se não existir
        if not collection_exists:
            schema = {
                "class": "Article",
                "description": "Artigos da base de conhecimento Movidesk",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text"
                    }
                },
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "resumo", "dataType": ["text"]},
                    {"name": "url", "dataType": ["text"]},
                    {"name": "status", "dataType": ["text"]},
                    {"name": "movidesk_id", "dataType": ["int"]},
                    {"name": "createdDate", "dataType": ["date"]},
                    {"name": "updatedDate", "dataType": ["date"]}
                ]
            }
            client.collections.create(schema)
            logger.info("✨ Schema criado com sucesso")
            await asyncio.sleep(1)  # Aguardar criação ser processada
        else:
            logger.info("ℹ️ Collection 'Article' já existe")
            
    except Exception as e:
        logger.error(f"❌ Erro ao criar schema: {str(e)}")
        raise

async def importar_artigos_movidesk(progress_callback=None, reset_base: bool = True):
    """Função principal de importação."""
    logger.info("🚀 Iniciando importação de artigos do Movidesk")
    
    try:
        client = get_weaviate_client()
        await verificar_e_criar_schema(client, reset_base=reset_base)
        
        total_importados = 0
        total_atualizados = 0
        pagina = 1

        # Get collection reference
        collection = client.collections.get("Article")
        
        while True:
            artigos = await buscar_lista_artigos(pagina)
            if not artigos:
                break

            for idx, artigo in enumerate(artigos, 1):
                try:
                    artigo_id = artigo.get("id")
                    if not artigo_id:
                        continue

                    # Verificar se já existe
                    if await verificar_artigo_existe(collection, artigo_id):
                        total_atualizados += 1
                        continue

                    # Buscar detalhes
                    detalhes = await buscar_detalhes_artigo(artigo_id)
                    if not detalhes or not detalhes.get("contentText"):
                        continue

                    # Gerar embedding
                    content = detalhes.get("contentText", "").strip()
                    embedding = await gerar_embedding_conteudo(content)
                    if not embedding:
                        continue

                    # Preparar dados
                    data_object = {
                        "movidesk_id": artigo_id,
                        "title": detalhes.get("title", ""),
                        "content": content,
                        "status": "Publicado",
                        "url": f"{os.getenv('BASE_ARTICLE_URL')}/{artigo_id}/{detalhes.get('slug', '')}",
                        "createdDate": from_iso_to_date(detalhes.get("createdDate")),
                        "updatedDate": from_iso_to_date(detalhes.get("updatedDate"))
                    }

                    # Inserir diretamente na collection
                    collection.data.insert(
                        data_object,
                        vector=embedding
                    )
                    
                    total_importados += 1
                    logger.info(f"✅ Artigo importado: {data_object['title']}")

                    # Progress callback
                    if progress_callback:
                        await progress_callback({
                            "status": "in_progress",
                            "pagina_atual": pagina,
                            "artigo_atual": idx,
                            "total_artigos": len(artigos),
                            "total_importados": total_importados,
                            "total_atualizados": total_atualizados,
                            "ultimo_artigo": data_object['title']
                        })

                    await asyncio.sleep(0.5)  # Rate limiting

                except Exception as e:
                    logger.error(f"❌ Erro ao processar artigo: {str(e)}")
                    continue

            pagina += 1

        # Finalizar progresso
        if progress_callback:
            await progress_callback({
                "status": "completed",
                "total_importados": total_importados,
                "total_atualizados": total_atualizados,
                "mensagem": "Importação concluída com sucesso!"
            })

        return {
            "status": "success",
            "total_importados": total_importados,
            "total_atualizados": total_atualizados
        }
        
    except Exception as e:
        logger.error(f"❌ Erro durante importação: {str(e)}")
        if progress_callback:
            await progress_callback({
                "status": "error",
                "mensagem": f"Erro durante a importação: {str(e)}"
            })
        raise HTTPException(status_code=500, detail=str(e))

async def report_progress(callback, pagina, idx, total, importados, atualizados, ultimo_artigo):
    """Helper para relatório de progresso"""
    percentual = (idx / total) * 100
    progress = {
        "pagina_atual": pagina,
        "total_processado": importados + atualizados,
        "progresso_pagina": f"{idx}/{total}",
        "percentual": f"{percentual:.1f}%",
        "total_importados": importados,
        "total_atualizados": atualizados,
        "ultimo_artigo": ultimo_artigo,
        "status": f"Processando página {pagina}"
    }
    await callback(progress)

@router.post("/importar-artigos")
async def endpoint_importar(reset_base: bool = False):
    """Endpoint para iniciar importação com opção de resetar a base."""
    async def progress_callback(progress):
        logger.info(f"📊 Progresso: {progress}")
        # Aqui você pode implementar lógica para enviar o progresso ao front-end, como WebSocket ou SSE.

    return await importar_artigos_movidesk(progress_callback=progress_callback, reset_base=reset_base)
