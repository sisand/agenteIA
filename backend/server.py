from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.clients import connect_clients, close_clients
from backend.routers import api

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    try:
        print("🔄 Inicializando conexões com serviços externos...")
        await connect_clients()
        print("✅ Todas as conexões inicializadas com sucesso.")
        yield
    except Exception as e:
        print(f"❌ Erro fatal na inicialização: {str(e)}")
        raise
    finally:
        print("🔒 Encerrando conexões...")
        await close_clients()
        print("✅ Conexões encerradas com sucesso.")

# Inicializar FastAPI com configurações explícitas
app = FastAPI(
    title="Agente de Suporte Sisand API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar todas as rotas através do router central
app.include_router(api.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
