import openai
from rasa_sdk import Action
from rasa_sdk.events import UserUtteranceReverted
import os

# Substituir a chave fixa pelo carregamento do .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


class ActionFallbackGPT(Action):
    def name(self):
        return "action_fallback_gpt"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get(
            "text"
        )  # Captura a pergunta do usuário

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente de suporte técnico da Sisand. Responda apenas perguntas relacionadas aos produtos da Sisand, como Vision ERP, Market Connector, Sisand Academy e outros sistemas da empresa. Se a pergunta não estiver relacionada, diga que não pode ajudar.",
                },
                {"role": "user", "content": user_message},
            ],
        )

        resposta = response.choices[0].message.content

        dispatcher.utter_message(resposta)

        return [UserUtteranceReverted()]  # Evita que a conversa fique bagunçada
