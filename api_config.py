"""
Configurações centralizadas das APIs
"""

# Base URLs
API_BASE = "http://127.0.0.1:8000"

# Endpoints
class APIEndpoints:
    # Chat e IA
    ASK = f"{API_BASE}/ask"
    PROMPT = f"{API_BASE}/prompt"
    METRICS = f"{API_BASE}/metrics"
    EMBEDDINGS = f"{API_BASE}/embeddings"
    FEEDBACK = f"{API_BASE}/feedback"
    
    # Curadoria
    TICKETS = f"{API_BASE}/movidesk-tickets"
    WEAVIATE_SAVE = f"{API_BASE}/weaviate-save"
    WEAVIATE_ARTIGOS = f"{API_BASE}/weaviate-artigos"
    CURADORIA = f"{API_BASE}/curadoria"
    CURADORIA_TICKETS = f"{API_BASE}/curadoria/tickets-curados"
    CURADORIA_GPT = f"{API_BASE}/gpt-curadoria"
    
    # Sessões
    SESSOES = f"{API_BASE}/sessoes"
