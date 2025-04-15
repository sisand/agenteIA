from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from supabase import create_client
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

# Configurar conexão com o Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)


class ActionBuscarArtigo(Action):
    def name(self):
        return "action_buscar_artigo"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")
        articles = (
            supabase.table("articles")
            .select("*")
            .ilike("content", f"%{query}%")
            .execute()
        )

        if articles.data:
            resposta = "Encontrei os seguintes artigos:\n"
            for art in articles.data:
                resposta += f"- [{art['title']}]({art['url']})\n"
        else:
            resposta = "Desculpe, não encontrei artigos relacionados."

        dispatcher.utter_message(text=resposta)
        return []
