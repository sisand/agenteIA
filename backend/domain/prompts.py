from fastapi import APIRouter, HTTPException
from backend.clients import get_supabase_client
import logging
from typing import List
from pydantic import BaseModel

router = APIRouter()

class PromptResponse(BaseModel):
    nome: str
    conteudo: str
    ativo: bool = True

@router.get("/ativos")
async def listar_prompts_ativos():
    """Lista todos os prompts ativos."""
    try:
        supabase = get_supabase_client()
        result = (
            supabase.table("prompts")
            .select("*")
            .eq("ativo", True)
            .execute()
        )
        # Garantir que retorna uma lista mesmo que vazia
        return result.data if result.data else []
    except Exception as e:
        logging.error(f"Erro ao listar prompts: {e}")
        return []

@router.get("/{nome}")
async def buscar_prompt(nome: str):
    """Busca um prompt específico pelo nome."""
    try:
        supabase = get_supabase_client()
        result = (
            supabase.table("prompts")
            .select("*")
            .eq("nome", nome)
            .eq("ativo", True)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Prompt não encontrado")
        # Retornar o primeiro item do resultado
        return result.data[0]
    except Exception as e:
        logging.error(f"Erro ao buscar prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{nome}")
async def atualizar_prompt(nome: str, prompt: PromptResponse):
    """Atualiza um prompt existente."""
    try:
        supabase = get_supabase_client()
        result = (
            supabase.table("prompts")
            .update({"conteudo": prompt.conteudo})
            .eq("nome", nome)
            .execute()
        )
        return {"success": True}
    except Exception as e:
        logging.error(f"Erro ao atualizar prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))
