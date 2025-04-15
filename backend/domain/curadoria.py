from fastapi import APIRouter, HTTPException
from backend.domain.models import Curadoria
import json
import os
from datetime import datetime
from app.clients import get_supabase_client

router = APIRouter()

CURADORIA_FILE = "curadorias.json"

def listar_curadorias():
    """
    Lista todas as curadorias armazenadas no arquivo JSON.
    """
    if not os.path.exists(CURADORIA_FILE):
        return []
    with open(CURADORIA_FILE, "r") as f:
        return json.load(f)

def registrar_curadoria(curadoria: Curadoria):
    """
    Registra uma nova curadoria no arquivo JSON.
    """
    curadoria_data = {
        "ticket_id": curadoria.ticket_id,
        "curador": curadoria.curador,
        "question": curadoria.question,
        "answer": curadoria.answer,
        "data": datetime.now().isoformat(),
    }

    if not os.path.exists(CURADORIA_FILE):
        with open(CURADORIA_FILE, "w") as f:
            json.dump([], f)

    with open(CURADORIA_FILE, "r") as f:
        curadorias = json.load(f)

    if any(c["ticket_id"] == curadoria.ticket_id for c in curadorias):
        raise HTTPException(status_code=400, detail="Este ticket já foi curado.")

    curadorias.append(curadoria_data)
    with open(CURADORIA_FILE, "w") as f:
        json.dump(curadorias, f, indent=2)

    return {"message": "Curadoria registrada com sucesso ✅"}

@router.get("/")
def get_curadorias():
    """
    Lista todas as curadorias.
    """
    try:
        return listar_curadorias()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
def post_curadoria(payload: Curadoria):
    """
    Registra uma nova curadoria.
    """
    try:
        return registrar_curadoria(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/acoes-ticket/{ticket_id}")
def get_acoes_ticket(ticket_id: int):
    """
    Retorna as ações realizadas em um ticket específico.
    """
    supabase = get_supabase_client()
    try:
        response = supabase.table("acoes_tickets").select("*").eq("ticket_id", ticket_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar ações do ticket: {e}")

@router.post("/curadoria")
async def registrar_curadoria(payload: Curadoria):
    """
    Registra uma nova curadoria.
    """
    try:
        # Simulação de registro
        return {"status": "Curadoria registrada com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao registrar curadoria: {str(e)}")
