from app.clients import get_supabase_client 
from supabase_utils import carregar_palavras_sociais


def carregar_social_keywords():
    try:
        supabase = get_supabase_client()
        response = (
            supabase.table("parametros")
            .select("valor")
            .eq("nome", "social_keywords")
            .execute()
        )
        if response.data and response.data[0].get("valor"):
            keywords_csv = response.data[0]["valor"]
            palavras = [k.strip().lower() for k in keywords_csv.split(",") if k.strip()]
            if palavras:
                return palavras
    except Exception as e:
        print(f"Erro ao buscar social_keywords: {e}")  # 👈 log visível

    # fallback padrão
    return [
        "oi",
        "olá",
        "bom dia",
        "boa tarde",
        "boa noite",
        "tudo bem",
        "como vai",
        "e aí",
        "quem é você",
        "seu nome",
        "você é real",
        "você é humano",
        "me conta sobre você",
        "como você está",
    ]


def eh_pergunta_social(pergunta: str) -> bool:
    """Verifica se uma pergunta é social baseada nas palavras-chave armazenadas"""
    try:
        palavras_chave = carregar_palavras_sociais()
        if not palavras_chave:
            return False
            
        palavras_lista = [p.strip().lower() for p in palavras_chave.split(',')]
        pergunta_lower = pergunta.lower()
        
        return any(palavra in pergunta_lower for palavra in palavras_lista)
    except Exception as e:
        print(f"❌ Erro ao verificar pergunta social: {e}")
        return False


def responder_direto_sem_rag(pergunta: str, user_id: str = "amigo") -> str:
    nome = user_id or "amigo"
    prompt = f"""
Você é um assistente virtual acolhedor da Sisand. O nome do usuário é {nome}.
Responda com empatia e simpatia. Não inclua informações técnicas ou da base de conhecimento.
Pergunta: {pergunta}
"""
    resposta = openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.7
    )
    return resposta.choices[0].message.content.strip()


def gerar_resposta_social(pergunta: str, nome_usuario: str = "amigo") -> str:
    """Gera uma resposta amigável para perguntas sociais"""
    return f"Olá {nome_usuario}! Sou o Sisandinho, um assistente virtual. Como posso ajudar com dúvidas sobre o sistema?"