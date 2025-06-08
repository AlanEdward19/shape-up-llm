# Shape-Up LLM

Este projeto utiliza dois modelos de machine learning para análise de postura e geração de insights a partir de anamneses, voltados para profissionais de nutrição e treinamento.

## Modelos

### 1. Posture Model (`posture_model`)
- **Função:** Analisa desvios posturais a partir de imagens.
- **Exemplo de uso:** Veja o arquivo `main.py`.
- **Arquivo de teste:** Uma imagem de exemplo está disponível em `files/Escoliose.jpg`.
- **Como usar:**
  - Importe o módulo `posture_model`.
  - Utilize a função de análise passando o caminho da imagem.

### 2. Insights Model (`insights_model`)
- **Função:** Lê arquivos de anamnese em formato CSV e gera insights relevantes para profissionais de nutrição e treinamento.
- **Parâmetro `type`:** Define o tipo de insight gerado (ex: nutrição, treinamento).
- **Exemplo de uso:** Veja o arquivo `main.py`.
- **Arquivo de teste:** Um exemplo de CSV está disponível em `files/sample.csv`.
- **Como usar:**
  - Importe o módulo `insights_model`.
  - Utilize a função de geração de insights passando o caminho do CSV e o parâmetro `type`.
  - **Importante:** Para executar o `insights_model`, copie o arquivo `.env.example` para `.env` e preencha os valores necessários (por exemplo, a chave da API do GPT).

## Estrutura do Projeto

```
main.py                # Exemplos de uso dos modelos
README.md              # Este arquivo
requirements.txt       # Dependências do projeto
.env.example           # Exemplo de variáveis de ambiente necessárias
files/                 # Arquivos de teste (imagem e CSV)
insights_model/        # Código do modelo de insights
posture_model/         # Código do modelo de postura
```

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. (Somente para o insights_model) Copie o arquivo `.env.example` para `.env` e preencha os valores necessários.
3. Execute o arquivo principal:
   ```bash
   python main.py
   ```

## Observações
- Os exemplos de uso para ambos os modelos estão implementados em `main.py`.
- Adapte os caminhos dos arquivos conforme necessário.

---

Desenvolvido para auxiliar profissionais de saúde e bem-estar na análise de postura e interpretação de anamneses.

