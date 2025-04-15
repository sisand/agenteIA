from fastapi import APIRouter, HTTPException
from backend.domain.models import FeedbackAberto

router = APIRouter()

# Salvar um novo feedback
@router.post("/feedback")
async def salvar_feedback(payload: FeedbackAberto):
    """
    Salva um novo feedback.
    """
    try:
        # Simulação de salvamento
        return {"status": "Feedback salvo com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(e)}")

# Listar feedbacks existentes
@router.get("/feedback")
async def listar_feedbacks():
    """
    Lista todos os feedbacks existentes.
    """
    try:
        # Simulação de listagem
        feedbacks = [
            {"id": 1, "comentario": "Muito bom!", "tipo": "positivo"},
            {"id": 2, "comentario": "Precisa melhorar.", "tipo": "negativo"},
        ]
        return feedbacks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar feedbacks: {str(e)}")
