from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage
from app.langchain_chain import qa_chain

router = APIRouter()

@router.post("")
async def ask_langchain_route(request: Request):
    body = await request.json()
    question = body.get("question")
    history = body.get("history", [])

    # Por enquanto, usamos só a pergunta. O histórico pode ser usado depois.
    messages = [HumanMessage(content=question)]

    result = qa_chain.invoke({"question": question, "chat_history": history})
    return {"answer": result["answer"]}
