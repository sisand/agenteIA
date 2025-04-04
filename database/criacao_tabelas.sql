-- Certifique-se de que a extensão pgvector está habilitada

CREATE EXTENSION IF NOT EXISTS vector;
-- Tabela para armazenar as mensagens do usuário e do assistente


drop table if exists mensagens;
CREATE TABLE mensagens (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    usuario_id text NOT NULL,
    pergunta text NOT NULL,
    resposta text NOT NULL,
    embedding vector,  -- tipo USER-DEFINED (extensão pgvector)
    created_at timestamptz DEFAULT now()
);

create index idx_mensagens_usuario on mensagens(usuario_id);


-- Remove a tabela mensagens (se já existir)
drop table if exists mensagens;

-- Tabela de sessões
create table sessoes (
    id uuid primary key default gen_random_uuid(),
    usuario_id text not null,
    inicio timestamptz default now(),
    fim timestamptz
);

create index idx_sessoes_usuario on sessoes(usuario_id);

-- Tabela de mensagens com relação à sessão
create table mensagens (
    id uuid primary key default gen_random_uuid(),
    sessao_id uuid references sessoes(id),
    usuario_id text not null,
    pergunta text not null,
    resposta text not null,
    embedding vector,  -- pgvector
    created_at timestamptz default now()
);

-- Índices para otimizar consultas
create index idx_mensagens_usuario on mensagens(usuario_id);
create index idx_mensagens_sessao on mensagens(sessao_id);






-- Tabela para armazenar feedbacks das respostas
CREATE TABLE IF NOT EXISTS feedbacks (
    id SERIAL PRIMARY KEY,
    pergunta TEXT NOT NULL,
    resposta TEXT NOT NULL,
    comentario TEXT,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Tabela de usuários (identificados pelo login do workspace)
CREATE TABLE usuarios (
    id SERIAL PRIMARY KEY,
    login VARCHAR(100) UNIQUE NOT NULL,
    nome VARCHAR(100),
    criado_em TIMESTAMP DEFAULT NOW()
);

# --- SCRIPT SQL PARA SUPABASE ---

CREATE TABLE prompts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  nome text NOT NULL,
  descricao text,
  conteudo text NOT NULL,
  ativo boolean DEFAULT true,
  criado_em timestamp with time zone DEFAULT now(),
  atualizado_em timestamp with time zone DEFAULT now()
);

-- Exemplo de insert inicial (pode ser feito via Supabase Studio)
INSERT INTO prompts (nome, descricao, conteudo)
VALUES (
  'padrao',
  'Prompt principal utilizado pelo assistente da Sisand',
  $$
  Você é um assistente da Sisand, que atua com acolhimento, didática e foco em orientar.

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

  Não repita trechos idênticos dos artigos. Reescreva com leveza, clareza e foco em orientar.
  $$
);

-- Curadoria
INSERT INTO prompts (nome, descricao, conteudo, ativo)
VALUES (
  'curadoria',
  'Prompt usado na geração estruturada de curadorias via GPT',
  'Você é um especialista em suporte técnico de sistemas ERP para concessionárias.

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
}
',
  true
);

create table if not exists parametros (
  nome text primary key,
  valor text not null,
  atualizado_em timestamp with time zone default timezone('utc'::text, now())
);


C-- Script de criação da função match_mensagens para Supabase (PostgreSQL + pgvector)

DROP FUNCTION IF EXISTS match_mensagens(vector, double precision);

CREATE OR REPLACE FUNCTION match_mensagens(
    embedding_input vector,
    min_similarity float8
)
RETURNS TABLE (
    id uuid,
    usuario_id text,
    pergunta text,
    resposta text,
    similarity float8,
    created_at timestamptz  -- tipo atualizado e correto
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id, 
        m.usuario_id, 
        m.pergunta, 
        m.resposta, 
        1 - (m.embedding <=> embedding_input) AS similarity,
        m.created_at  -- tipo compatível
    FROM mensagens AS m
    WHERE (1 - (m.embedding <=> embedding_input)) > min_similarity
    ORDER BY similarity DESC
    LIMIT 10;
END;
$$ LANGUAGE plpgsql;




