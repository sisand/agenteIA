from datetime import datetime, timedelta

def get_brazil_time():
    """Retorna datetime atual no fuso horário do Brasil (UTC-3)."""
    return datetime.utcnow() - timedelta(hours=3)

def gerar_embedding_openai(texto: str):
    """Simula a geração de embeddings usando OpenAI."""
    return [0.1, 0.2, 0.3]  # Exemplo de vetor
