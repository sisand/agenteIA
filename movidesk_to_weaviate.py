from weaviate import connect_to_wcs
from weaviate.auth import AuthApiKey
from weaviate.util import generate_uuid5
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
import requests
import time
from datetime import datetime
import tiktoken
import os

# ======================= CONFIGS =======================
# Carregar chaves do .env
MOVI_TOKEN = os.getenv("MOVI_TOKEN")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_CLASS = "Article"
ZERAR_BASE = True

HEADERS = {"Content-Type": "application/json"}
BASE_URL_LIST = "https://api.movidesk.com/public/v1/kb/article"
BASE_URL_SINGLE = "https://api.movidesk.com/public/v1/article"
BASE_ARTICLE_URL = "https://sisand.movidesk.com/kb/pt-br/article"
PAGE_SIZE = 50

# ======================= CONVERSORES =======================
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

def contar_tokens(texto, model="gpt-3.5-turbo"):
    if not texto:
        return 0
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(texto))

# ======================= CONECTA AO WEAVIATE =======================
client = connect_to_wcs(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY)
)
print("✅ Conectado ao Weaviate.")

# ======================= VERIFICAR/CRIAR COLEÇÃO =======================
collections = client.collections.list_all()

if ZERAR_BASE and WEAVIATE_CLASS in collections:
    client.collections.delete(WEAVIATE_CLASS)
    print(f"🗑️ Coleção '{WEAVIATE_CLASS}' deletada.")

if WEAVIATE_CLASS not in collections:
    client.collections.create(
        name=WEAVIATE_CLASS,
        description="Artigos do Movidesk",
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="contenthtml", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="status", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="createdDate", data_type=DataType.DATE),
            Property(name="updatedDate", data_type=DataType.DATE),
            Property(name="url", data_type=DataType.TEXT, skip_vectorization=True),
        ],
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        )
    )
    print(f"✅ Coleção '{WEAVIATE_CLASS}' criada com sucesso.")
else:
    print(f"✅ Coleção '{WEAVIATE_CLASS}' já existe.")

collection = client.collections.get(WEAVIATE_CLASS)

# ======================= BUSCA E ENVIO DOS ARTIGOS =======================
all_articles = []
page = 0
print("🔍 Buscando artigos no Movidesk...")

while True:
    params = {"token": MOVI_TOKEN, "$top": PAGE_SIZE, "page": page}
    resp = requests.get(BASE_URL_LIST, headers=HEADERS, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"⚠️ Erro ao buscar artigos na página {page}: {resp.text}")
        break

    data = resp.json()
    items = data.get("items", [])
    if not items:
        break

    all_articles.extend(items)
    print(f"✔️ Página {page} com {len(items)} artigos.")
    page += 1
    time.sleep(0.5)

print(f"📚 Total de artigos encontrados: {len(all_articles)}")

# ======================= ENVIAR PARA WEAVIATE =======================
print("🚀 Enviando artigos para o Weaviate...")

for kb_art in all_articles:
    article_id = kb_art["id"]
    print(f"\n➡️ Processando artigo ID {article_id} - {kb_art.get('title', '')[:50]}")
    
    single_url = f"{BASE_URL_SINGLE}/{article_id}"
    r = requests.get(single_url, headers=HEADERS, params={"token": MOVI_TOKEN})
    if r.status_code != 200:
        print(f"❌ Falha ao buscar artigo completo ID {article_id}")
        continue
    full = r.json()

    content = full.get("contentText", "").strip()
    if not content:
        print(f"⚠️ Artigo ID {article_id} sem conteúdo textual.")
        continue

    html = full.get("contentHtml") or full.get("contenthtml") or ""
    obj = {
        "title": full.get("title", ""),
        "content": content,
        "contenthtml": html,
        "status": traduz_status(full.get("articleStatus", 0)),
        "createdDate": from_iso_to_date(full.get("createdDate")),
        "updatedDate": from_iso_to_date(full.get("updatedDate")),
        "url": f"{BASE_ARTICLE_URL}/{article_id}/{full.get('slug', '')}"
    }

    uid = generate_uuid5(str(full["id"]))
    try:
        if collection.data.exists(uuid=uid):
            collection.data.update(uuid=uid, properties=obj)
            print(f"♻️ Artigo '{obj['title'][:40]}...' atualizado com sucesso.")
        else:
            collection.data.insert(uuid=uid, properties=obj)
            print(f"✅ Artigo '{obj['title'][:40]}...' inserido com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao enviar '{obj['title'][:40]}': {e}")

client.close()
print("🎉 Finalizado!")
