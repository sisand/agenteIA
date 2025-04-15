import sys
import os
import weaviate
from weaviate.collections.classes.config import DataType
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.clients import connect_clients

def atualizar_schema():
    # Conectar e obter cliente
    connect_clients()
    from backend.clients import weaviate_client  # Importar após conectar
    
    if weaviate_client is None:
        raise RuntimeError("❌ Falha ao conectar com Weaviate")
    
    try:
        # Verificar e deletar collection existente
        if hasattr(weaviate_client, 'collections') and weaviate_client.collections.exists("Article"):
            weaviate_client.collections.delete("Article")
            print("✅ Collection antiga removida")
    except Exception as e:
        print(f"⚠️ Erro ao deletar collection: {e}")
    
    try:
        weaviate_client.collections.create(
            name="Article",
            properties=[
                {
                    "name": "article_id",
                    "data_type": DataType.INT,
                    "description": "ID único do artigo (autoincremental)",
                    "autoIncrement": True
                },
                {
                    "name": "id_movidesk",
                    "data_type": DataType.INT,
                    "description": "ID do artigo no Movidesk",
                    "indexFilterable": True,
                    "indexSearchable": True
                },
                {
                    "name": "title",
                    "data_type": DataType.TEXT,
                    "description": "Título do artigo"
                },
                {
                    "name": "content",
                    "data_type": DataType.TEXT,
                    "description": "Conteúdo completo do artigo"
                },
                {
                    "name": "resumo",
                    "data_type": DataType.TEXT,
                    "description": "Resumo do artigo"
                },
                {
                    "name": "status",
                    "data_type": DataType.TEXT,
                    "description": "Status do artigo"
                },
                {
                    "name": "createdDate",
                    "data_type": DataType.DATE,
                    "description": "Data de criação"
                },
                {
                    "name": "updatedDate",
                    "data_type": DataType.DATE,
                    "description": "Data de atualização"
                },
                {
                    "name": "url",
                    "data_type": DataType.TEXT,
                    "description": "URL do artigo"
                }
            ],
            vectorizer_config=None
        )
        print("✅ Schema atualizado com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao criar schema: {e}")
        raise e
    finally:
        # Garantir que a conexão seja fechada
        if weaviate_client:
            weaviate_client.close()

if __name__ == "__main__":
    atualizar_schema()
