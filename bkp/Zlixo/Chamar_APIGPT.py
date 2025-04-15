import openai
import os

# Substitua a chave secreta pelo carregamento da variável de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Qual é o maior mamífero do mundo?"}],
    )
    print(response.choices[0].message.content)
except openai.AuthenticationError as e:
    print("Erro de autenticação:", e)
