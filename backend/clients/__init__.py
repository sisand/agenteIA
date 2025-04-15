import os
from typing import Optional
from openai import OpenAI
from supabase import create_client, Client
from weaviate import connect_to_wcs
from weaviate.auth import AuthApiKey

# Variáveis globais
_weaviate_client = None
_openai_client = None
_supabase_client = None

def get_weaviate_client():
    global _weaviate_client
    if not _weaviate_client:
        try:
            _weaviate_client = connect_to_wcs(
                cluster_url=os.getenv("WEAVIATE_URL"),
                auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
            )

            if _weaviate_client.is_ready():
                print("✅ Conexão com Weaviate estabelecida")
            else:
                raise Exception("Weaviate não está pronto")
        except Exception as e:
            print(f"Erro ao inicializar Weaviate: {str(e)}")
            raise

    return _weaviate_client

    return _weaviate_client
def get_openai_client():
    """Retorna cliente OpenAI inicializado."""
    global _openai_client
    if not _openai_client:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def get_supabase_client() -> Optional[Client]:
    """Retorna cliente Supabase inicializado."""
    global _supabase_client
    if not _supabase_client:
        _supabase_client = create_client(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_KEY", "")
        )
    return _supabase_client

async def gerar_embedding_openai(texto: str) -> Optional[list]:
    """Gera embedding usando OpenAI."""
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texto
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None

async def connect_clients():
    """Inicializa conexões de forma assíncrona."""
    try:
        print("🔄 Conectando ao Supabase...")
        get_supabase_client()
        print("✅ Supabase conectado.")

        print("🔄 Conectando ao Weaviate...")
        get_weaviate_client()
        print("✅ Weaviate conectado.")

        print("🤖 Configurando OpenAI...")
        get_openai_client()
        print("✅ OpenAI configurado.")
    except Exception as e:
        print(f"❌ Erro ao conectar clientes: {e}")
        raise

async def close_clients():
    """Fecha conexões de forma assíncrona."""
    global _weaviate_client, _openai_client, _supabase_client
    _weaviate_client = None
    _openai_client = None
    _supabase_client = None
