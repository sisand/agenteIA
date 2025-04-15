import os
import logging
from dotenv import load_dotenv
from supabase import create_client
import weaviate
from weaviate.classes.init import Auth
from openai import OpenAI
from config import OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY, SUPABASE_URL, SUPABASE_KEY

# 🧪 Carrega variáveis de ambiente
load_dotenv()

# 🔐 Variáveis globais (não inicializar aqui!)
supabase_client = None
_weaviate_client = None
_openai_client = None

# 🔍 Validação das variáveis de ambiente obrigatórias
if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("As variáveis SUPABASE_URL e SUPABASE_KEY são obrigatórias.")
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise EnvironmentError("As variáveis WEAVIATE_URL e WEAVIATE_API_KEY são obrigatórias.")
if not OPENAI_API_KEY:
    raise EnvironmentError("A variável OPENAI_API_KEY é obrigatória.")

# 🔗 Função para conectar todos os clients necessários
def connect_clients():
    """
    Inicializa as conexões com Supabase, OpenAI e Weaviate.
    """
    global supabase_client, _weaviate_client, _openai_client

    if not supabase_client:
        print("📦 Conectando ao Supabase...")
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase conectado.")

    try:
        print("🔄 Conectando ao Weaviate...")
        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={
                "X-OpenAI-Api-Key": OPENAI_API_KEY
            }
        )
        
        if _weaviate_client.is_ready():
            print("✅ Weaviate conectado.")
        else:
            raise RuntimeError("Weaviate não está pronto")

        print("🤖 Configurando OpenAI...")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI configurado.")
        
    except Exception as e:
        print(f"❌ Erro ao conectar clientes: {str(e)}")
        raise

# 🛠️ Função para verificar se os clientes estão conectados
def ensure_clients_connected():
    """
    Garante que todos os clientes necessários estão conectados.
    """
    if not supabase_client or not _weaviate_client or not _openai_client:
        connect_clients()

# 🔒 Encerrar conexões
def close_clients():
    """
    Fecha conexões com serviços externos.
    """
    global _weaviate_client, _openai_client
    print("🛑 Fechando conexões...")
    _weaviate_client = None
    _openai_client = None

# 🧰 Getters seguros
def get_weaviate_client():
    """
    Retorna uma instância conectada do cliente Weaviate Cloud.
    """
    if not _weaviate_client:
        raise RuntimeError("Weaviate client não está conectado")
    return _weaviate_client

def get_openai_client():
    """
    Retorna o cliente OpenAI configurado.
    """
    if not _openai_client:
        raise RuntimeError("OpenAI client não está conectado")
    return _openai_client

def get_supabase_client():
    """
    Retorna o cliente Supabase conectado.
    """
    if not supabase_client:
        raise RuntimeError("Supabase client não está conectado.")
    return supabase_client

def gerar_embedding_openai(texto: str):
    """
    Gera embedding para um texto usando a API da OpenAI.
    """
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            input=texto,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {str(e)}")
        raise RuntimeError(f"Erro ao gerar embedding: {str(e)}")

