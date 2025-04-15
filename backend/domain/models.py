from pydantic import BaseModel
from typing import List, Optional

class Curadoria(BaseModel):
    ticket_id: int
    curador: str
    question: str
    answer: str

class AskRequest(BaseModel):
    question: str
    usuario_nome: str
    usuario_id: str
    personalidade: str

class FeedbackAberto(BaseModel):
    comentario: str
    tipo: str
    usuario_id: int

class ArticleInput(BaseModel):
    title: str
    content: str
    url: str = "#"
    summary: str = ""
    type: str = "resposta_ticket"

class Pergunta(BaseModel):
    question: str
    personalidade: str = ""
    use_gpt: bool = True
    limit: int = 3
    modelo: str = "gpt-3.5-turbo"
    usuario_login: str = "anonimo"
    usuario_nome: Optional[str] = None
    historico: List[dict] = []
    prompt_sistema: Optional[str] = None  # Novo campo

class Resposta(BaseModel):
    resposta: str
    artigos: List[dict] = []
    tempo: float = 0
    prompt_usado: Optional[str] = None  # Adicionar campo para o prompt usado

class PromptTemplate(BaseModel):
    nome: str
    conteudo: str
    ativo: bool = True
