from fastapi import APIRouter

router = APIRouter()

@router.get("/metrics")
async def obter_metricas():
    """
    Retorna métricas do sistema.
    """
    return {
        "total_perguntas": 100,
        "tempo_medio_resposta": 1.23,
        "feedbacks_recebidos": 50
    }
