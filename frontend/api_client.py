import requests

API_URL = "http://127.0.0.1:8000"  # Base URL do backend

def enviar_pergunta(pergunta, usuario_nome, usuario_id, modelo, personalidade="padrao", prompt_sistema=None, historico=None):
    """
    Envia uma pergunta para o backend e retorna a resposta.
    """
    if historico is None:
        historico = []

    payload = {
        "question": pergunta,
        "personalidade": personalidade,
        "use_gpt": True,
        "limit": 3,
        "modelo": modelo,
        "usuario_login": usuario_id,
        "usuario_nome": usuario_nome,
        "historico": historico,
        "prompt_sistema": prompt_sistema,  # Adicionando o prompt do sistema
    }
    try:
        # Certifique-se de que o endpoint inclui o prefixo correto
        response = requests.post(f"{API_URL}/chat/ask", json=payload, timeout=10)
        response.raise_for_status()  # Levanta uma exceção para códigos de status HTTP de erro
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"resposta": f"Erro ao conectar com a API: {e}"}
    except ValueError:
        return {"resposta": "Erro ao processar a resposta da API. Verifique o formato do JSON."}

def carregar_feedbacks():
    response = requests.get(f"{API_URL}/feedbacks")
    return response.json()

def carregar_parametros():
    response = requests.get(f"{API_URL}/metrics")
    return response.json()

def carregar_historico(usuario_id):
    response = requests.get(f"{API_URL}/conversas", params={"usuario_id": usuario_id})
    return response.json()

def buscar_tickets(limite):
    response = requests.get(f"{API_URL}/movidesk-tickets", params={"limite": limite})
    return response.json()

def salvar_curadoria(ticket_id, curador, question, answer):
    payload = {"ticket_id": ticket_id, "curador": curador, "question": question, "answer": answer}
    response = requests.post(f"{API_URL}/curadoria", json=payload)
    return response.json()

def listar_artigos():
    response = requests.get(f"{API_URL}/weaviate-artigos")
    return response.json()

def carregar_prompts():
    """
    Carrega todos os prompts ativos.
    """
    try:
        response = requests.get(f"{API_URL}/prompts/ativos", timeout=5)
        if response.status_code == 200:
            # Garantir que retorna uma lista mesmo que vazia
            return response.json() or []
        return []
    except Exception as e:
        print(f"Erro ao carregar prompts: {e}")
        return []

def carregar_prompt_especifico(nome: str):
    """
    Carrega um prompt específico pelo nome.
    """
    try:
        response = requests.get(f"{API_URL}/prompts/{nome}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Retornar o primeiro item se for uma lista, ou o próprio objeto
            return data[0] if isinstance(data, list) else data
        return None
    except Exception as e:
        print(f"Erro ao carregar prompt específico: {e}")
        return None

def salvar_prompt(nome, conteudo):
    payload = {"nome": nome, "conteudo": conteudo}
    response = requests.post(f"{API_URL}/prompts", json=payload)
    return response.json()

def atualizar_prompt(nome: str, conteudo: str):
    """
    Atualiza o conteúdo de um prompt existente.
    """
    try:
        response = requests.put(
            f"{API_URL}/prompts/{nome}",
            json={"nome": nome, "conteudo": conteudo, "ativo": True},
            timeout=5
        )
        if response.status_code == 200:
            return {"success": True}
        return {"success": False, "error": f"Erro ao atualizar: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def carregar_sessoes(usuario_id):
    response = requests.get(f"{API_URL}/sessoes", params={"usuario_id": usuario_id})
    return response.json()

def carregar_embeddings():
    """
    Busca os embeddings armazenados no backend.
    """
    try:
        response = requests.get(f"{API_URL}/embeddings", timeout=10)
        response.raise_for_status()  # Levanta uma exceção para códigos de status HTTP de erro
        if response.text.strip():  # Verifica se a resposta não está vazia
            return response.json()
        else:
            return {"error": "Resposta vazia da API."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Erro ao conectar com a API: {e}"}
    except ValueError:
        return {"error": "Resposta da API não está em formato JSON válido."}

def importar_artigos_movidesk(reset_base: bool = False):
    """Importa artigos usando nova rota padronizada."""
    try:
        response = requests.post(
            f"{API_URL}/api/importar",
            json={"reset_base": reset_base},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Erro na importação: {str(e)}"}

def carregar_acoes_ticket(ticket_id):
    """
    Busca as ações realizadas em um ticket específico.
    """
    response = requests.get(f"{API_URL}/curadoria/acoes-ticket/{ticket_id}")
    return response.json()

def carregar_configuracoes():
    """
    Carrega todas as configurações do sistema.
    """
    try:
        response = requests.get(f"{API_URL}/configuracoes")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def atualizar_configuracao(nome: str, valor: str):
    """
    Atualiza uma configuração específica.
    """
    try:
        response = requests.post(
            f"{API_URL}/configuracoes",
            json={"nome": nome, "valor": valor}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}
