from app.inference import generate_insights
from app.utils import read_anamnese_csv

if __name__ == "__main__":
    print("Lendo anamnese de exemplo...")
    text_input = read_anamnese_csv("app/sample.csv")

    insights_nutri = generate_insights("nutricionist", text_input)
    print("---- INSIGHTS NUTRICIONISTA ----")
    print(insights_nutri)
