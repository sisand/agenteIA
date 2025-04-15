# supabase_utils.py
"""
Módulo centralizado para operações com Supabase
Contém todas as funções de acesso a dados e manipulação de entidades
"""

from datetime import datetime, timedelta
from app.clients import get_supabase_client

def get_brazil_time():
    """Retorna datetime atual no fuso horário do Brasil (UTC-3)"""
    return datetime.utcnow() - timedelta(hours=3)

def obter_ou_criar_usuario(login: str, nome: str = None) -> int:
    try:
        supabase = get_supabase_client()
        result = supabase.table("usuarios").select("id").eq("login", login).limit(1).execute()
        
        if result.data:
            return result.data[0]["id"]
            
        novo_usuario = supabase.table("usuarios").insert({
            "login": login,
            "nome": nome or login,
            "criado_em": get_brazil_time().isoformat()
        }).execute()
        
        return novo_usuario.data[0]["id"]
    except Exception as e:
        print(f"❌ Erro ao obter/criar usuário: {e}")
        raise e

def encerrar_sessao_antiga(sessao_id: int):
    try:
        supabase = get_supabase_client()
        supabase.table("sessoes").update({
            "fim": get_brazil_time().isoformat()
        }).eq("id", sessao_id).execute()
    except Exception as e:
        print(f"Erro ao encerrar sessão {sessao_id}: {e}")

def obter_ou_criar_sessao(usuario_id: int) -> int:
    try:
        supabase = get_supabase_client()
        print(f"🔍 Verificando sessão ativa para o usuário: {usuario_id}")
        
        resultado = supabase.table("sessoes").select("id, inicio").eq("usuario_id", usuario_id).is_("fim", "null").order("inicio", desc=True).limit(1).execute()

        if resultado.data:
            ultima_sessao = resultado.data[0]
            data_inicio = datetime.fromisoformat(ultima_sessao["inicio"])
            
            if get_brazil_time() - data_inicio < timedelta(minutes=10):
                return ultima_sessao["id"]
            
            encerrar_sessao_antiga(ultima_sessao["id"])

        nova_sessao = supabase.table("sessoes").insert({
            "usuario_id": usuario_id,
            "inicio": get_brazil_time().isoformat()
        }).execute()
        
        return nova_sessao.data[0]["id"]

    except Exception as e:
        print(f"❌ Erro ao obter ou criar sessão: {e}")
        raise e

def salvar_mensagem(usuario_id: int, pergunta: str, resposta: str, embedding: list = None, sessao_id: int = None):
    try:
        if not sessao_id:
            raise ValueError("Sessão não definida para salvar a mensagem.")

        payload = {
            "usuario_id": usuario_id,
            "sessao_id": sessao_id,
            "pergunta": pergunta,
            "resposta": resposta,
            "embedding": embedding,
            "created_at": get_brazil_time().isoformat()
        }

        supabase = get_supabase_client()
        supabase.table("mensagens").insert(payload).execute()
        print("✅ Mensagem salva com sucesso.")

    except Exception as e:
        print(f"❌ Erro ao salvar mensagem: {e}")
        raise e

def busca_mensagens_por_embedding(embedding, min_similarity=0.1):
    try:
        embedding_list = list(embedding)

        print(f"🔍 Enviando para Supabase com o embedding: {embedding_list[:10]}")
        print(f"min_similarity: {min_similarity}")

        # Chamada da função match_mensagens
        response = supabase.rpc(
            "match_mensagens",
            {"embedding_input": embedding_list, "min_similarity": min_similarity},
        ).execute()

        print(f"📦 Dados brutos retornados do Supabase: {response.data}")

        mensagens = []
        if response.data:
            for msg in response.data:
                similarity = msg.get("similarity", 0)
                if similarity >= min_similarity:
                    mensagens.append(
                        {
                            "id": msg.get("id"),
                            "usuario_id": msg.get("usuario_id"),
                            "pergunta": msg.get("pergunta"),
                            "resposta": msg.get("resposta"),
                            "similarity": round(similarity * 100, 2),
                            "data": msg.get("created_at", ""),
                        }
                    )

        return {"dados": mensagens}
    except Exception as e:
        print(f"❌ Erro ao buscar mensagens: {e}")
        return {"dados": []}

# Operações de Feedback
def salvar_feedback(pergunta: str, resposta: str, comentario: str, tipo: str = None, usuario_id: int = None):
    """Salva um feedback no Supabase com todos os campos necessários"""
    try:
        supabase = get_supabase_client()
        data = {
            "pergunta": pergunta,
            "resposta": resposta,
            "comentario": comentario,
            "tipo": tipo,
            "usuario_id": usuario_id,
            "criado_em": get_brazil_time().isoformat()  # Usando horário do Brasil
        }
        result = supabase.table("feedbacks").insert(data).execute()
        print(f"✅ Feedback #{result.data[0]['id']} salvo com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar feedback: {e}")
        raise e

# Operações de Prompt
def buscar_prompt(nome: str = "padrao") -> str:
    """Busca um prompt específico no Supabase"""
    try:
        supabase = get_supabase_client()
        result = (
            supabase.table("prompts")
            .select("nome, conteudo, descricao")  # Selecionando todos os campos relevantes
            .eq("nome", nome)
            .eq("ativo", True)
            .limit(1)
            .execute()
        )

        if result.data:
            return result.data[0]["conteudo"]
        return "Prompt não encontrado."
    except Exception as e:
        print(f"❌ Erro ao buscar prompt: {e}")
        return "Erro ao buscar prompt."

# Operações de Parâmetros IA
def carregar_parametros_ia() -> dict:
    """Carrega todos os parâmetros da IA do Supabase"""
    try:
        supabase = get_supabase_client()
        dados = supabase.table("parametros").select("nome", "valor").execute()
        return {item["nome"]: item["valor"] for item in dados.data}
    except Exception as e:
        print(f"Erro ao carregar parâmetros: {e}")
        return {}

def salvar_parametros_ia(modelo: str, temperatura: float, top_p: float):
    """Salva os parâmetros da IA no Supabase"""
    try:
        supabase = get_supabase_client()
        supabase.table("parametros").upsert([
            {"nome": "modelo", "valor": modelo},
            {"nome": "temperatura", "valor": str(temperatura)},
            {"nome": "top_p", "valor": str(top_p)},
        ]).execute()
    except Exception as e:
        print(f"Erro ao salvar parâmetros: {e}")
        raise e

def salvar_parametro(nome: str, valor: str):
    """Salva um único parâmetro no Supabase seguindo o schema correto"""
    try:
        supabase = get_supabase_client()
        data = {
            "nome": nome,
            "valor": valor,
            "atualizado_em": get_brazil_time().isoformat()
        }
        supabase.table("parametros").upsert(data, on_conflict="nome").execute()
        print(f"✅ Parâmetro {nome} salvo com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar parâmetro {nome}: {e}")
        raise e

def buscar_ultimas_mensagens(usuario_id: str, limite=3):
    try:
        response = (
            supabase.table("mensagens")
            .select("pergunta, resposta")
            .eq("usuario_id", usuario_id)
            .order("criado_em", desc=True)
            .limit(limite)
            .execute()
        )
        mensagens = response.data or []
        return mensagens
    except Exception as e:
        print(f"❌ Erro ao buscar mensagens: {e}")
        return []

def salvar_palavras_sociais(palavras: str) -> bool:
    """Salva as palavras-chave sociais na tabela de parâmetros"""
    try:
        supabase = get_supabase_client()
        data = {
            "nome": "social_keywords",
            "valor": palavras.strip(),
            "atualizado_em": get_brazil_time().isoformat()
        }
        supabase.table("parametros").upsert(data, on_conflict="nome").execute()
        print("✅ Palavras sociais atualizadas com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar palavras sociais: {e}")
        return False

def carregar_palavras_sociais() -> str:
    """Carrega as palavras-chave sociais da tabela de parâmetros"""
    try:
        supabase = get_supabase_client()
        result = (
            supabase.table("parametros")
            .select("valor")
            .eq("nome", "social_keywords")
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]["valor"]
        return ""
    except Exception as e:
        print(f"❌ Erro ao carregar palavras sociais: {e}")
        return ""