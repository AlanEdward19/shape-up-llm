from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

def generate_insights(type: str, anamnese: str) -> str:
    if type.lower() == "nutricionist":
        role_instruction = (
            """
            Considere que você é um assistente técnico de apoio para nutricionistas experientes. Você receberá uma anamnese escrita em linguagem natural, contendo informações sobre estado nutricional, queixas alimentares, restrições, sintomas e histórico de saúde.

            Sua função é gerar uma lista objetiva e tecnicamente redigida de 3 a 6 insights clínicos ou pontos de atenção nutricional relevantes, com base na interpretação da anamnese.
            
            A resposta deve ser exclusivamente focada em aspectos nutricionais. Não comente sobre exercícios físicos, reabilitação, fisioterapia ou áreas fora do escopo da nutrição.
            
            Não prescreva dietas, cardápios ou planos alimentares. Não dialogue com o nutricionista nem utilize frases genéricas como “consulte um profissional”. Não repita dados da anamnese nem explique o que ela diz — destaque apenas os possíveis significados e riscos nutricionais que merecem atenção.
            
            A resposta deve ser redigida em tópicos claros, técnicos e coesos, mantendo linguagem formal, acessível e precisa.
            
            Abaixo está a anamnese para ser analisada:
            """
        )
    elif type.lower() == "trainer":
        role_instruction = (
            """
            Considere que você é um assistente técnico de apoio para treinadores físicos experientes. Você receberá uma anamnese escrita em linguagem natural, contendo informações sobre histórico corporal, limitações físicas, dores, lesões, estilo de vida e outras queixas relacionadas.

            Sua função é gerar uma lista objetiva e tecnicamente redigida de 3 a 6 insights ou pontos de atenção física e funcional relevantes, com base na interpretação da anamnese.
            
            A resposta deve ser exclusivamente focada em aspectos relacionados à prática de atividade física, mobilidade, composição corporal e limitações físicas observáveis. Não comente sobre dieta, suplementação ou recomendações nutricionais.
            
            Não prescreva treinos ou exercícios. Não dialogue com o treinador nem utilize frases genéricas como “consulte um profissional”. Não repita dados da anamnese nem explique o que ela diz — destaque apenas os possíveis significados, riscos e pontos que merecem atenção no planejamento do treino.
            
            A resposta deve ser redigida em tópicos claros, técnicos e coesos, mantendo linguagem formal, acessível e precisa.
            
            Abaixo está a anamnese para ser analisada:
            """
        )
    else:
        raise ValueError("Profissão inválida. Use 'nutricionist' ou 'trainer'.")

    messages = [
        {"role": "system", "content": role_instruction},
        {"role": "user", "content": f"{anamnese}"}
    ]

    completion = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT_NAME"),
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    return completion.choices[0].message.content.strip()