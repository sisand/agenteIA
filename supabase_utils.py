# supabase_utils.py
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# supabase_utils.py
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def salvar_parametros_ia(modelo: str, temperatura: float, top_p: float):
    try:
        parametros = {
            "modelo": modelo,
            "temperatura": str(temperatura),
            "top_p": str(top_p)
        }
        for nome, valor in parametros.items():
            supabase.table("parametros").upsert({"nome": nome, "valor": valor}).execute()
        print("✅ Parâmetros IA salvos com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar parâmetros IA: {e}")

def carregar_parametros_ia():
    try:
        response = supabase.table("parametros").select("nome", "valor").execute()
        if response.data:
            return {item["nome"]: item["valor"] for item in response.data}
        return {}
    except Exception as e:
        print(f"❌ Erro ao carregar parâmetros IA: {e}")
        return {}


def busca_mensagens_por_embedding(embedding, min_similarity=0.1):
    try:
        embedding_list = list(embedding)

        print(f"🔍 Enviando para Supabase com o embedding: {embedding_list[:10]}")
        print(f"min_similarity: {min_similarity}")

        # Chamada da função match_mensagens
        response = supabase.rpc('match_mensagens', {
            'embedding_input': embedding_list,
            'min_similarity': min_similarity
        }).execute()

        print(f"📦 Dados brutos retornados do Supabase: {response.data}")

        mensagens = []
        if response.data:
            for msg in response.data:
                similarity = msg.get('similarity', 0)
                if similarity >= min_similarity:
                    mensagens.append({
                        'id': msg.get('id'),
                        'usuario_id': msg.get('usuario_id'),
                        'pergunta': msg.get('pergunta'),
                        'resposta': msg.get('resposta'),
                        'similarity': round(similarity * 100, 2),
                        'data': msg.get('created_at', '')
                    })

        return {'dados': mensagens}
    except Exception as e:
        print(f"❌ Erro ao buscar mensagens: {e}")
        return {'dados': []}

def salvar_mensagem(usuario_id: str, pergunta: str, resposta: str, embedding: list):
    try:
        response = supabase.table("mensagens").insert({
            "usuario_id": usuario_id,
            "pergunta": pergunta,
            "resposta": resposta,
            "embedding": embedding
        }).execute()
        print("✅ Mensagem salva com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar mensagem: {e}")
        return False



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


def salvar_feedback(pergunta: str, resposta: str, comentario: str):
    try:
        data = {
            "pergunta": pergunta,
            "resposta": resposta,
            "comentario": comentario,
            "criado_em": datetime.utcnow().isoformat()
        }
        supabase.table("feedbacks").insert(data).execute()
        print("✅ Feedback salvo com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar feedback: {e}")
