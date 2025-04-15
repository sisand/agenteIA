import os
from supabase import create_client
from weaviate import Client as WeaviateClient
from openai import OpenAI

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def get_weaviate_client():
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")
    return WeaviateClient(url, api_key)

def get_openai_client():
    # Simulação de cliente OpenAI
    return None

def connect_clients():
    print("Conectando aos clientes...")

def close_clients():
    print("Fechando conexões...")
