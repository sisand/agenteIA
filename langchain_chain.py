from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_weaviate import WeaviateVectorStore
from app.clients import weaviate_client, openai_client

qa_chain = None  # 🔹 Inicializa como None

def criar_qa_chain():
    global qa_chain

    # Importa dinamicamente para garantir que pega o valor atualizado
    from app.clients import weaviate_client, openai_client

    if weaviate_client is None:
        raise RuntimeError("❌ Erro: o cliente do Weaviate não está inicializado.")

    retriever = WeaviateVectorStore(
        client=weaviate_client,
        index_name="Article",
        text_key="content"
    ).as_retriever()

    llm = ChatOpenAI(api_key=openai_client.api_key)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
