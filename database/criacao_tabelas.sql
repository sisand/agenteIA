-- Extensões necessárias
CREATE EXTENSION IF NOT EXISTS vector;

-- Remover tabelas existentes
DROP TABLE IF EXISTS mensagens CASCADE;
DROP TABLE IF EXISTS sessoes CASCADE;
DROP TABLE IF EXISTS feedbacks CASCADE;
DROP TABLE IF EXISTS usuarios CASCADE;
DROP TABLE IF EXISTS prompts CASCADE;
DROP TABLE IF EXISTS parametros CASCADE;

-- Tabela de usuários
CREATE TABLE usuarios (
    id SERIAL PRIMARY KEY,
    login VARCHAR(100) UNIQUE NOT NULL,
    nome VARCHAR(100),
    criado_em TIMESTAMP DEFAULT now()
);

-- Tabela de sessões
CREATE TABLE sessoes (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuarios(id),
    inicio TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fim TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_sessoes_usuario ON sessoes(usuario_id);

-- Tabela de mensagens
CREATE TABLE mensagens (
    id SERIAL PRIMARY KEY,
    sessao_id INTEGER REFERENCES sessoes(id) ON DELETE CASCADE,
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    pergunta TEXT NOT NULL,
    resposta TEXT NOT NULL,
    embedding VECTOR,
    classificacao_huggingface TEXT,
    pontuacao_classificacao NUMERIC,
    pipeline_usado TEXT,
    acao_langchain TEXT,
    tipo_resposta TEXT,
    contexto_ativo TEXT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_mensagens_usuario ON mensagens(usuario_id);
CREATE INDEX idx_mensagens_sessao ON mensagens(sessao_id);
CREATE INDEX idx_classificacao ON mensagens(classificacao_huggingface);
CREATE INDEX idx_tipo_resposta ON mensagens(tipo_resposta);

-- Tabela de feedbacks
CREATE TABLE feedbacks (
    id SERIAL PRIMARY KEY,
    pergunta TEXT NOT NULL,
    resposta TEXT NOT NULL,
    comentario TEXT,
    tipo TEXT,
    usuario_id INTEGER REFERENCES usuarios(id),
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de prompts
CREATE TABLE prompts (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL,
    descricao TEXT,
    conteudo TEXT NOT NULL,
    ativo BOOLEAN DEFAULT TRUE,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prompts padrão
INSERT INTO prompts (nome, descricao, conteudo)
VALUES (
  'padrao',
  'Prompt principal utilizado pelo assistente da Sisand',
  $$Você é um assistente da Sisand, que atua com acolhimento, didática e foco em orientar.

Histórico recente da conversa:
{historico_texto}

Abaixo estão alguns artigos técnicos sobre o sistema da Sisand:
{context}

Com base nesses artigos e no histórico da conversa, responda de forma clara e útil à seguinte pergunta:

{question}

✅ Estruture a resposta com:
- Saudação inicial e acolhimento.
- Passo a passo claro e detalhado.
- Explique o objetivo ou a importância de cada passo.
- Dica prática no final.
- Encerramento simpático e com canal de suporte.

Não repita trechos idênticos dos artigos. Reescreva com leveza, clareza e foco em orientar.$$
);

INSERT INTO prompts (nome, descricao, conteudo)
VALUES (
  'curadoria',
  'Prompt usado na geração estruturada de curadorias via GPT',
  $$Você é um especialista em suporte técnico de sistemas ERP para concessionárias.

Com base no resumo a seguir, gere uma estrutura formal com os seguintes campos:

1. **Pergunta do cliente**: Reescreva com clareza e tom consultivo, como se fosse enviada por um cliente.
2. **Solução aplicada**: Explique o que foi feito em passos práticos e objetivos.
3. **Diagnóstico** (opcional): Indique a possível causa raiz, se for possível.
4. **Resultado final** (opcional): Descreva o que aconteceu após aplicar a solução.
5. **Dica adicional** (opcional): Sugestão para evitar o problema ou facilitar o uso.

Mantenha um tom consultivo, fluido e padronizado, como se fosse publicado numa base de conhecimento da equipe técnica.

Resumo do atendimento:
\"\"\"{texto}\"\"\"

Responda apenas neste formato JSON:
{
  "pergunta": "...",
  "solucao": "...",
  "diagnostico": "...",
  "resultado": "...",
  "dica": "..."
}$$
);

-- Tabela de parâmetros
CREATE TABLE parametros (
    nome TEXT PRIMARY KEY,
    valor TEXT NOT NULL,
    atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Função para busca semântica
DROP FUNCTION IF EXISTS match_mensagens(vector, double precision);

CREATE OR REPLACE FUNCTION match_mensagens(
    embedding_input VECTOR,
    min_similarity FLOAT8
)
RETURNS TABLE (
    id INTEGER,
    usuario_id INTEGER,
    pergunta TEXT,
    resposta TEXT,
    similarity FLOAT8,
    created_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id, 
        m.usuario_id, 
        m.pergunta, 
        m.resposta, 
        1 - (m.embedding <=> embedding_input) AS similarity,
        m.created_at
    FROM mensagens AS m
    WHERE (1 - (m.embedding <=> embedding_input)) > min_similarity
    ORDER BY similarity DESC
    LIMIT 10;
END;
$$ LANGUAGE plpgsql;

-- Adicionar parâmetros para prompts padrão
INSERT INTO parametros (nome, valor) VALUES
('prompt_chat_padrao', 'padrao'),
('prompt_curadoria_padrao', 'curadoria');

