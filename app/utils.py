import pandas as pd

def read_anamnese_csv(file_path: str) -> str:
    df = pd.read_csv(file_path, delimiter=";")
    text = ""
    for col in df.columns:
        text += f"{col}: {df[col][0]}\n"
    return text