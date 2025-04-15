## Lista de Funcionalidades do Projeto

| **#** | **Funcionalidade**                                   | **Descrição**                                                                                    |
|-------|------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1     | Controle de palavras sociais                         | Configura palavras-chave sociais no Supabase e identifica se uma pergunta é social ou técnica.   |
| 2     | Controle de gravação no Supabase                     | Grava sessões, usuários, feedbacks, mensagens e parâmetros no Supabase.                          |
| 3     | Usuário recebendo dados do Workspace                 | Simula o recebimento de dados do Workspace e grava o usuário no Supabase.                        |
| 4     | Menu "Base de Conhecimento"                          | Inclui submenus para importar artigos e visualizar embeddings.                                   |
| 5     | Pergunta no chat com embedding e busca no Weaviate   | Embeda a pergunta do usuário e realiza uma busca semântica no Weaviate.                          |
| 6     | Prompt gravado no Supabase                           | Lê e edita prompts padrão e de curadoria no Supabase.                                            |
| 7     | Artigos com melhor similaridade                      | Retorna artigos com a melhor similaridade da busca semântica.                                    |
| 8     | Prompt para melhor resultado                         | Utiliza um prompt para apresentar os resultados da busca semântica de forma clara ao usuário.    |
| 9     | Busca de artigos somente quando necessário           | Identifica se a pergunta é social ou técnica usando Hugging Face e realiza a busca somente se necessário. |
| 10    | Login                                                | Solicita login quando o parâmetro `HABILITAR_LOGIN` está ativado.                                |
| 11    | Leitura e edição de prompts                          | Permite ler e editar prompts padrão e de curadoria no Supabase.                                  |
| 12    | Renderização das ações dos tickets                   | Renderiza as ações dos tickets, incluindo mensagens e interações.                                |
| 13    | Importação de artigos do Movidesk                    | Importa artigos do Movidesk para o Weaviate.                                                     |
| 14    | Estilo de resposta                                   | Permite selecionar o estilo de resposta: técnico, professor ou atendente simpática.              |
| 15    | Chat com layout moderno, OCR e feedback              | Renderiza o chat com layout moderno, suporta OCR e permite feedback.                             |
| 16    | Visualização de feedbacks                            | Exibe feedbacks recebidos e permite exportar para CSV.                                           |
| 17    | Painel administrativo                                | Exibe métricas do sistema, como total de perguntas e tempo médio de resposta.                    |
| 18    | Curadoria de tickets                                 | Permite realizar curadoria de tickets resolvidos.                                                |
| 19    | Conversas finalizadas                                | Exibe o histórico de conversas finalizadas.                                                      |
| 20    | Encerramento e criação de nova sessão                | Encerra e cria uma nova sessão automaticamente após 10 minutos de inatividade.                   |

---

## Lista de APIs do Projeto

| **Método** | **Rota**                          | **Descrição**                                |
|------------|-----------------------------------|----------------------------------------------|
| POST       | `/ask_langchain`                  | Ask Langchain Route                          |
| POST       | `/ask`                            | Ask Question                                 |
| GET        | `/metrics`                        | Get Metrics                                  |
| POST       | `/feedback`                       | Salvar Feedback                              |
| GET        | `/embeddings`                     | Listar Embeddings                            |
| GET        | `/curadoria`                      | Listar Curadorias                            |
| POST       | `/curadoria`                      | Registrar Curadoria                          |
| GET        | `/curadoria/tickets-curados`      | Listar Tickets Curados                       |
| POST       | `/gpt-curadoria`                  | Gerar Curadoria Com Gpt                      |
| POST       | `/search-only`                    | Buscar Apenas Sem Gpt                        |
| GET        | `/movidesk-tickets`               | Buscar Tickets Movidesk                      |
| GET        | `/weaviate-artigos`               | Listar Artigos Weaviate                      |
| POST       | `/weaviate-artigos`               | Inserir Artigo Weaviate                      |
| GET        | `/prompt`                         | Get Prompt                                   |
| POST       | `/prompt`                         | Salvar Prompt                                |
| GET        | `/conversas`                      | Listar Conversas                             |
| GET        | `/sessoes`                        | Listar Sessoes                               |
