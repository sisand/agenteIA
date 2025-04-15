import logging
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from fastapi import APIRouter, HTTPException
from backend.clients import get_weaviate_client, get_openai_client
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
from pydantic import BaseModel

class ImportacaoRequest(BaseModel):
    reset_base: bool = False

router = APIRouter()
logger = logging.getLogger(__name__)

def from_iso_to_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None

async def buscar_lista_artigos(pagina: int = 1, limite: int = 50) -> List[Dict[str, Any]]:
    try:
        token = os.getenv("MOVI_TOKEN")
        base_url = os.getenv("MOVI_LIST_URL")

        params = {
            "token": token,
            "page": pagina,
            "$top": limite,
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
        return data.get("items", [])

    except Exception as e:
        logger.error(f"❌ Erro ao buscar lista de artigos: {str(e)}")
        return []

async def buscar_todos_artigos_movidesk() -> List[Dict[str, Any]]:
    artigos = []
    pagina = 1
    PAGE_SIZE = 30

    while True:
        logger.info(f"🔄 Buscando artigos na página {pagina}...")
        lista = await buscar_lista_artigos(pagina=pagina, limite=PAGE_SIZE)
        if not lista:
            break
        artigos.extend(lista)
        if len(lista) < PAGE_SIZE:
            break
        pagina += 1

    logger.info(f"📚 Total de artigos recuperados: {len(artigos)}")
    return artigos

async def importar_artigo(client, artigo: Dict[str, Any]) -> bool:
    try:
        article_id = artigo.get("id")
        token = os.getenv("MOVI_TOKEN")
        url = f"{os.getenv('MOVI_DETAIL_URL')}/{article_id}"

        response = requests.get(
            url,
            params={"token": token},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        article = response.json()

        content = article.get("contentText", "").strip()
        if not content:
            return False

        openai_client = get_openai_client()

        # Gerar resumo com GPT
        resumo_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Resuma o texto abaixo em 3 a 5 linhas claras e objetivas:"},
                {"role": "user", "content": content}
            ],
            temperature=0.5
        )
        resumo = resumo_response.choices[0].message.content.strip()

        # Embedar o resumo, não o conteúdo completo
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=resumo
        )

        data_object = {
            "title": article.get("title"),
            "content": content,
            "resumo": resumo,
            "movidesk_id": article_id,
            "status": "Publicado",
            "url": f"{os.getenv('BASE_ARTICLE_URL')}/{article_id}/{article.get('slug')}",
            "createdDate": from_iso_to_date(article.get("createdDate")),
            "updatedDate": from_iso_to_date(article.get("updatedDate"))
        }

        collection = client.collections.get("Article")
        uuid = generate_uuid5(str(article_id))

        if collection.data.exists(uuid=uuid):
            collection.data.update(uuid=uuid, properties=data_object, vector=embedding_response.data[0].embedding)
            logger.info(f"🔁 Artigo atualizado: {article_id}")
        else:
            collection.data.insert(uuid=uuid, properties=data_object, vector=embedding_response.data[0].embedding)
            logger.info(f"✅ Artigo inserido: {article_id}")

        return True

    except Exception as e:
        logger.error(f"❌ Erro ao importar artigo {artigo.get('id')}: {str(e)}")
        return False

async def verificar_e_criar_schema(client, reset_base: bool = True):
    try:
        collection_exists = any(col.name == "Article" for col in client.collections.list_all().values())

        if collection_exists and reset_base:
            logger.info(f"✅ Collection 'Article' detectada: {collection_exists}")
            try:
                client.collections.delete("Article")
                logger.info("🗑️ Collection 'Article' excluída com sucesso")
                await asyncio.sleep(2)
                collection_exists = False
            except Exception as e:
                logger.error(f"❌ Erro ao excluir collection: {str(e)}")
                raise

        if not collection_exists:
            client.collections.create(
                name="Article",
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="resumo", data_type=DataType.TEXT),
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="status", data_type=DataType.TEXT),
                    Property(name="movidesk_id", data_type=DataType.INT),
                    Property(name="createdDate", data_type=DataType.DATE),
                    Property(name="updatedDate", data_type=DataType.DATE),
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small",
                    vectorize_collection_name=False
                )
            )
            logger.info("✨ Schema criado com sucesso")
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"❌ Erro ao criar schema: {str(e)}")
        raise

async def importar_artigos_movidesk(progress_callback=None, reset_base: bool = True):
    client = get_weaviate_client()
    logger.info("🔄 Verificando e configurando schema no Weaviate...")
    await verificar_e_criar_schema(client, reset_base=reset_base)

    artigos = await buscar_todos_artigos_movidesk()
    importados = 0

    for idx, artigo in enumerate(artigos):
        if await importar_artigo(client, artigo):
            importados += 1

        if progress_callback:
            await progress_callback({
                "status": "in_progress",
                "pagina": (idx // 50) + 1,
                "artigo_atual": idx + 1,
                "total_artigos": len(artigos),
                "total_importados": importados
            })
        await asyncio.sleep(0.5)

    if progress_callback:
        await progress_callback({
            "status": "completed",
            "total_importados": importados
        })

    logger.info(f"✅ Importação concluída! Total de artigos importados: {importados}")
    return {"total_importados": importados}

@router.post("/")
async def endpoint_importar(req: ImportacaoRequest):
    async def progress_callback(progress):
        logger.info(f"📊 Progresso: {progress}")

    logger.info(f"🚩 reset_base recebido do front: {req.reset_base}")

    try:
        return await importar_artigos_movidesk(
            progress_callback=progress_callback,
            reset_base=req.reset_base
        )
    except Exception as e:
        logger.error(f"❌ Erro na importação: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
