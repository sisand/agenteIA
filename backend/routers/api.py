from fastapi import APIRouter
from backend.domain import (
    chat,
    curadoria,
    feedbacks,
    metrics,
    prompts,
    importar_artigos
)

router = APIRouter()

# Rotas centralizadas com prefixos padronizados
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(curadoria.router, prefix="/curadoria", tags=["curadoria"])
router.include_router(feedbacks.router, prefix="/feedbacks", tags=["feedbacks"])
router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
router.include_router(prompts.router, prefix="/prompts", tags=["prompts"])
router.include_router(importar_artigos.router, prefix="/importar", tags=["importacao"])