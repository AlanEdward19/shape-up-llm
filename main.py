from insights_model.inference import generate_insights
from insights_model.utils import read_anamnese_csv
from posture_model.posture_analyzer import analyze_image

if __name__ == "__main__":
    #print("Lendo anamnese de exemplo...")
    #text_input = read_anamnese_csv("files/sample.csv")

    #insights_nutri = generate_insights("nutricionist", text_input)
    #print("---- INSIGHTS NUTRICIONISTA ----")
    #print(insights_nutri)
    #print("----------------------------------")
    #print("/n")

    #insights_nutri = generate_insights("trainer", text_input)
    #print("---- INSIGHTS Treinador ----")
    #print(insights_nutri)
    #print("----------------------------------")

    images = {
        "Costas": "files/Escoliose.jpg"
    }

    all_insights = []

    for name, path in images.items():
        insights = analyze_image(name, path)
        all_insights.extend(insights)

    print("\n--- INSIGHTS POSTURAIS ---")
    for insight in all_insights:
        print(insight)
