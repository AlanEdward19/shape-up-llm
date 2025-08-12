from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from posture_model.posture_analyzer import PostureAnalyzer
from insights_model.inference import generate_insights
from insights_model.utils import read_anamnese_csv
import cv2
import numpy as np

app = FastAPI()

@app.post("/analyze_posture")
async def analyze_posture(views: List[str] = Form(...), files: List[UploadFile] = File(...)):
    analyzer = PostureAnalyzer(static_image_mode=True)
    results = []

    if len(views) == 1 and "," in views[0]:
        views = [v.strip() for v in views[0].split(",")]

    for view, file in zip(views, files):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res = analyzer.analyze(img, view)
        results.append(res)
    return {
        "patientId": "",
        "professionalId": "",
        "servicePlanId": "",
        "createdAt": datetime.now().isoformat(),
        "images": results
    }

@app.post("/generate_insights")
async def generate_insights_endpoint(role: str, file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.csv", "wb") as f:
        f.write(contents)
    text_input = read_anamnese_csv("temp.csv")
    insights = generate_insights(role, text_input)
    return {"insights": insights}