import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from frontend.api_client import (
    enviar_pergunta,
    carregar_feedbacks,
    carregar_parametros,
    carregar_historico,
    buscar_tickets,
    salvar_curadoria,
    listar_artigos,
    salvar_prompt,
    carregar_prompts,
    carregar_sessoes,
    carregar_embeddings,
    importar_artigos_movidesk,
    carregar_acoes_ticket,
    carregar_configuracoes,
    atualizar_configuracao,
    atualizar_prompt,  # Adicionar esta linha
    carregar_prompt_especifico,
)
from datetime import datetime
from PIL import Image
import pytesseract
import requests

st.set_page_config(page_title="Agente de Suporte Sisand", layout="wide")

# === Configurações Iniciais ===
if "historico" not in st.session_state:
    st.session_state.historico = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "grupo_ativo" not in st.session_state:
    st.session_state.grupo_ativo = "🔎 Busca & IA"
if "submenu_ativo" not in st.session_state:
    st.session_state.submenu_ativo = "Chat Inteligente"

# === Estrutura do Menu com Submenus ===
menu_grupos = {
    "🔎 Busca & IA": {"Chat Inteligente": "chat"},
    "📥 Curadoria": {
        "Curadoria de Tickets": "curadoria_tickets",
        "Curadoria Manual": "curadoria_manual",
    },
    "📊 Gestão": {
        "Painel de Gestão": "painel_gestao",
        "Conversas Finalizadas": "conversas_finalizadas",
        "Feedbacks Recebidos": "feedbacks",
    },
    "📥 Base de Conhecimento": {
        "Importar Artigos": "importar_artigos",
        "Ver Embeddings": "ver_embeddings",
    },
    "⚙️ Configurações": {"Editar Prompt do Agente": "editar_prompt"},
}

# === Função para Renderizar o Menu ===
def renderizar_menu():
    # Renderizar o menu principal
    st.sidebar.title("Navegação")
    novo_grupo = st.sidebar.radio(
        "Escolha a seção:",
        list(menu_grupos.keys()),
        index=list(menu_grupos.keys()).index(st.session_state.grupo_ativo),
    )

    # Atualizar o submenu ao mudar o grupo
    if novo_grupo != st.session_state.grupo_ativo:
        st.session_state.grupo_ativo = novo_grupo
        st.session_state.submenu_ativo = list(menu_grupos[novo_grupo].keys())[0]

    # Renderizar o submenu
    novo_submenu = st.sidebar.radio(
        "Escolha o modo:",
        list(menu_grupos[st.session_state.grupo_ativo].keys()),
        index=list(menu_grupos[st.session_state.grupo_ativo].keys()).index(
            st.session_state.submenu_ativo
        ),
    )

    # Atualizar o submenu ativo
    st.session_state.submenu_ativo = novo_submenu

    # Retornar o modo selecionado
    return menu_grupos[st.session_state.grupo_ativo][st.session_state.submenu_ativo]

# === Determinar o Modo Ativo ===
modo_ativo = renderizar_menu()

# === Renderização de Acordo com o Modo Ativo ===
if modo_ativo == "chat":
    st.subheader("💬 Chat com IA")
    
    # Campo de pergunta com estilo melhorado
    pergunta = st.text_input(
        "Digite sua dúvida:",
        placeholder="Como posso ajudar você hoje?",
        key="chat_input"
    )
    
    # Botão de envio centralizado
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        enviar = st.button("Enviar pergunta", use_container_width=True)
    
    if enviar and pergunta:
        with st.spinner("Elaborando resposta..."):
            resposta = enviar_pergunta(
                pergunta=pergunta,
                usuario_nome="Usuário",
                usuario_id="anonimo",
                modelo="gpt-3.5-turbo"
            )
            
            if "resposta" in resposta:
                # Container para a resposta
                with st.container():
                    st.markdown("---")
                    st.markdown(f"**Sisandinho:** {resposta['resposta']}")
                    
                    # Artigos relacionados
                    if resposta.get("artigos"):
                        st.markdown("### 📎 Artigos Relacionados")
                        for artigo in resposta["artigos"]:
                            st.markdown(f"- [{artigo['title']}]({artigo['url']})")
                    
                    # Prompt utilizado com formatação adequada
                    with st.expander("🔍 Prompt utilizado na resposta", expanded=False):
                        if resposta.get("prompt_usado"):
                            st.markdown("### Prompt do Sistema")
                            st.text_area(
                                label="Prompt Completo",
                                value=resposta["prompt_usado"],
                                height=400,
                                disabled=True,
                                key="prompt_display",
                                label_visibility="collapsed"
                            )
                        else:
                            st.warning("Prompt não disponível")
                    st.markdown("---")
            else:
                st.error("Erro ao buscar resposta. Verifique o backend.")

elif modo_ativo == "curadoria_tickets":
    st.subheader("🎫 Buscar Tickets do Movidesk")
    limite = st.number_input("Quantidade de tickets a buscar", min_value=1, max_value=50, value=10)
    if st.button("Buscar Tickets"):
        tickets = buscar_tickets(limite)
        if tickets:
            for ticket in tickets.get("tickets", []):
                st.markdown(f"- **ID:** {ticket['id']} | **Assunto:** {ticket['subject']}")
                if st.button(f"Ver Ações do Ticket {ticket['id']}"):
                    acoes = carregar_acoes_ticket(ticket['id'])
                    if acoes:
                        st.markdown(f"### Ações do Ticket {ticket['id']}")
                        for acao in acoes:
                            st.markdown(f"- **Ação:** {acao['descricao']} | **Data:** {acao['data']}")
                    else:
                        st.info("Nenhuma ação encontrada para este ticket.")
        else:
            st.info("Nenhum ticket encontrado.")

elif modo_ativo == "curadoria_manual":
    st.subheader("📝 Curadoria Manual")
    ticket_id = st.text_input("ID do Ticket")
    curador = st.text_input("Nome do Curador")
    question = st.text_area("Pergunta do Cliente")
    answer = st.text_area("Resposta do Agente")
    if st.button("Salvar Curadoria"):
        if ticket_id and curador and question and answer:
            resultado = salvar_curadoria(ticket_id, curador, question, answer)
            st.success("Curadoria salva com sucesso!" if resultado else "Erro ao salvar curadoria.")
        else:
            st.warning("Preencha todos os campos antes de salvar.")

elif modo_ativo == "painel_gestao":
    st.subheader("📈 Métricas do Sistema")
    metrics = carregar_parametros()
    if metrics:
        st.metric("Total de Perguntas", metrics.get("total_perguntas", 0))
        st.metric("Tempo Médio de Resposta", metrics.get("tempo_medio_resposta", 0))
        st.metric("Feedbacks Recebidos", metrics.get("feedbacks_recebidos", 0))
    else:
        st.info("Nenhuma métrica encontrada.")

elif modo_ativo == "conversas_finalizadas":
    st.subheader("📜 Histórico de Conversas")
    try:
        sessoes = carregar_sessoes("anonimo")
        if sessoes:
            for sessao in sessoes:
                st.markdown(f"### Sessão: {sessao['sessao_id']}")
                for mensagem in sessao["mensagens"]:
                    st.markdown(f"- **Pergunta:** {mensagem['pergunta']}")
                    st.markdown(f"  **Resposta:** {mensagem['resposta']}")
        else:
            st.info("Nenhuma conversa encontrada.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao conectar com a API: {e}")
    except ValueError:
        st.error("Erro ao processar a resposta da API. Verifique o formato do JSON.")

elif modo_ativo == "feedbacks":
    st.subheader("📊 Feedbacks Recebidos")
    feedbacks = carregar_feedbacks()
    if feedbacks:
        for feedback in feedbacks:
            st.markdown(f"- **Comentário:** {feedback['comentario']} | **Tipo:** {feedback['tipo']}")
    else:
        st.info("Nenhum feedback encontrado.")

elif modo_ativo == "importar_artigos":
    st.subheader("📥 Importar Artigos do Movidesk")

    if "importacao_status" not in st.session_state:
        st.session_state.importacao_status = None

    reset_base = st.checkbox("Resetar base de dados", value=False)

    if st.button("Iniciar Importação"):
        with st.status("⏳ Iniciando importação...", expanded=True) as status:
            try:
                status.write("🚀 Enviando solicitação para o backend...")
                resposta = requests.post(
                    "http://localhost:8000/api/importar",
                    json={"reset_base": reset_base}
                )
                resultado = resposta.json()

                if resultado.get("total_importados", 0) > 0:
                    status.update(
                        label=f"✅ Importação finalizada com sucesso!",
                        state="complete"
                    )
                    st.success(f"🎉 Total de artigos importados: {resultado['total_importados']}")
                else:
                    status.update(
                        label="⚠️ Nenhum artigo foi importado.",
                        state="warning"
                    )
                    st.info("Verifique se já estavam na base ou se houve algum filtro.")

                st.session_state.importacao_status = {
                    "total_importados": resultado.get("total_importados", 0),
                    "reset_base": reset_base
                }

            except Exception as e:
                status.update(
                    label="❌ Falha durante a importação",
                    state="error"
                )
                st.error(f"Erro: {e}")

    # Exibe status da última importação
    if st.session_state.importacao_status:
        status = st.session_state.importacao_status
        with st.expander("📊 Último status da importação", expanded=False):
            st.markdown(f"- **Total importado:** {status.get('total_importados', 0)}")
            st.markdown(f"- **Base resetada:** {'Sim' if status.get('reset_base') else 'Não'}")


elif modo_ativo == "ver_embeddings":
    st.subheader("📊 Visualizar Embeddings")
    embeddings = carregar_embeddings()
    if "error" in embeddings:
        st.error(embeddings["error"])
    elif embeddings:
        st.write("Exibindo embeddings...")
        # Adicionar lógica de visualização de embeddings aqui.
    else:
        st.info("Nenhum embedding encontrado.")

elif modo_ativo == "editar_prompt":
    st.subheader("📝 Gerenciamento de Prompts")
    prompts = carregar_prompts()
    
    if not prompts:
        st.warning("Não foi possível carregar os prompts.")
    else:
        for prompt in prompts:
            with st.expander(f"Prompt: {prompt['nome']}"):
                conteudo = st.text_area(
                    f"Conteúdo do prompt",
                    value=prompt.get("conteudo", ""),
                    height=200
                )
                if st.button(f"Salvar alterações", key=f"save_{prompt['nome']}"):
                    result = salvar_prompt(prompt['nome'], conteudo)
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.success("Prompt atualizado com sucesso!")
