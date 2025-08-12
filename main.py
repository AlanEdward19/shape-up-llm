import json

import cv2

from insights_model.inference import generate_insights
from insights_model.utils import read_anamnese_csv
from posture_model.posture_analyzer import PostureAnalyzer

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

    analyzer = PostureAnalyzer(static_image_mode=True)
    img = cv2.imread("files/Direito.jpg")

    res = analyzer.analyze(img, "Right")
    a = json.dumps(res, ensure_ascii=False, indent=2)
    print(a)
