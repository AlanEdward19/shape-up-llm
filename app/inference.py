import torch


def generate_insight(text_input: str, tokenizer, model, type: str, max_new_tokens=1000):
    prompt = f"""
Você é um assistente especializado em nutrição clínica. Analise a anamnese abaixo e gere insights úteis e estruturados para o nutricionista. 
Baseie suas respostas nas informações fornecidas pelo paciente e gere recomendações práticas, sempre voltadas à nutrição e saúde alimentar.

Cada insight deve ser objetivo e apresentado em tópicos. Evite generalidades.

Exemplo:
- Paciente relatou intolerância à lactose → Sugerir evitar alimentos derivados do leite, como queijos e iogurtes comuns.
- Relatou pouca ingestão de vegetais → Orientar aumento de vegetais nas principais refeições.

Anamnese:
{text_input}

Insights nutricionais:
""" if type == "nutricionist" else f"""
Você é um assistente para treinadores experientes que querem insights claros e práticos a partir da anamnese de um cliente.

Leia a anamnese abaixo e gere somente uma lista curta (máximo 5 itens) com pontos de atenção diretamente relacionados ao treino.

**REGRAS:**
- Foco estrito nas condições do cliente que impactam a escolha dos exercícios.
- Use apenas nomes reais de exercícios se for necessário alertar sobre eles.
- Não faça recomendações genéricas, nem fale sobre dieta, avaliação médica ou planejamento geral.
- Não repita frases ou ideias.
- Evite linguagem vaga, confusa ou que pareça conversa.
- Responda em português formal, claro e objetivo.
- Não mencione nada sobre consultar outros profissionais, nem fale com o treinador (não use “você”, “deve”, etc).
- Dê só os insights práticos, pontuais, úteis para quem vai montar o treino.

---

Anamnese:
{text_input}

---

Exemplo de resposta para anamnese com hipertensão e intolerância à lactose (não presente aqui, só para modelo entender formato):

1. Evitar exercícios que causem picos abruptos de pressão arterial, como levantamento de peso máximo.
2. Priorizar exercícios aeróbicos de intensidade moderada.
3. Monitorar frequência cardíaca durante o treino.
4. Considerar exercícios de fortalecimento muscular que não sobrecarreguem as articulações.
5. Evitar agachamento com carga alta devido a possíveis limitações articulares.

---

Agora, gere os insights para o treino baseados na anamnese acima:
"""


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if type == "nutricionist":
        split_token = "Insights nutricionais:"
    else:
        split_token = "Agora, gere os insights para o treino baseados na anamnese acima:"
    if split_token in decoded:
        return decoded.split(split_token, 1)[-1].strip()
    return decoded.strip()
