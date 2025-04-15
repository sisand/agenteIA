import weaviate
import json

client = weaviate.Client(
    url="https://kegwrhvasmc0n279eqrqra.c0.us-west3.gcp.weaviate.cloud",  # Substitua
    auth_client_secret=weaviate.AuthApiKey(
        api_key="jRhoRcymKc4jbud6UJNPzlaE1TtQ6pPzeA0D"
    ),
    additional_headers={
        "X-OpenAI-Api-Key": "SUA_OPENAI_API_KEY"
    },  # Apenas se estiver usando modelo built-in
)

class_obj = {
    "class": "Article",
    "description": "Artigos da base de conhecimento do Movidesk",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {"model": "text-embedding-ada-002", "type": "text"}
    },
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "url", "dataType": ["text"]},
        {"name": "created", "dataType": ["text"]},
        {"name": "updated", "dataType": ["text"]},
    ],
}

# Cria a classe se ainda não existir
if not client.schema.contains({"class": "Article"}):
    client.schema.create_class(class_obj)

# Carrega os artigos do arquivo texto gerado
with open("movidesk_articles.txt", "r", encoding="utf-8") as file:
    articles = []
    current = {}
    content_lines = []
    for line in file:
        if line.startswith("ARTIGO ID:"):
            if current:
                current["content"] = "\n".join(content_lines).strip()
                articles.append(current)
            current = {"id": line.strip().split(":")[1].strip()}
            content_lines = []
        elif line.startswith("Título:"):
            current["title"] = line.strip().split(":", 1)[1].strip()
        elif line.startswith("Status:"):
            current["status"] = line.strip().split(":", 1)[1].strip()
        elif line.startswith("Criado:"):
            partes = line.split("|")
            current["created"] = partes[0].split(":")[1].strip()
            current["updated"] = partes[1].split(":")[1].strip()
        elif line.startswith("URL:"):
            current["url"] = line.strip().split(":", 1)[1].strip()
        elif line.startswith("=" * 80):
            continue
        else:
            content_lines.append(line)

    if current:
        current["content"] = "\n".join(content_lines).strip()
        articles.append(current)

# Envia os artigos para o Weaviate
for article in articles:
    client.data_object.create(
        data_object={
            "title": article.get("title"),
            "content": article.get("content"),
            "url": article.get("url"),
            "created": article.get("created"),
            "updated": article.get("updated"),
        },
        class_name="Article",
    )

print(f"✅ {len(articles)} artigos indexados no Weaviate com sucesso!")
