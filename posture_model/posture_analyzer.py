import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


def get_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    h, w = image.shape[:2]
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
    return landmarks


def check_shoulder_tilt(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    diff = abs(left_shoulder[1] - right_shoulder[1])
    return diff > 15, diff


def check_pelvic_tilt(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    diff = abs(left_hip[1] - right_hip[1])
    return diff > 15, diff


def check_lateral_spine_curve(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    mid_hip_x = (left_hip[0] + right_hip[0]) / 2
    diff = abs(mid_shoulder_x - mid_hip_x)
    return diff > 15, diff


def analyze_image(name, path):
    landmarks = get_landmarks(path)
    if not landmarks:
        return [f"[{name}] Nenhum corpo detectado na imagem."]

    insights = []

    if name == "Frente":
        scoliosis, scoliosis_diff = check_lateral_spine_curve(landmarks)
        if scoliosis:
            insights.append(f"[{name}] Possível escoliose (desvio lateral da coluna de {scoliosis_diff:.1f}px).")

        shoulder_tilt, s_diff = check_shoulder_tilt(landmarks)
        if shoulder_tilt:
            insights.append(f"[{name}] Possível tilt escapular (diferença de altura dos ombros de {s_diff:.1f}px).")

        pelvic_tilt, p_diff = check_pelvic_tilt(landmarks)
        if pelvic_tilt:
            insights.append(f"[{name}] Possível tilt pélvico (diferença de altura do quadril de {p_diff:.1f}px).")

    elif name in ["Lado Direito", "Lado Esquerdo"]:
        ear = landmarks[
            mp_pose.PoseLandmark.RIGHT_EAR.value if name == "Lado Direito" else mp_pose.PoseLandmark.LEFT_EAR.value]
        ankle = landmarks[
            mp_pose.PoseLandmark.RIGHT_ANKLE.value if name == "Lado Direito" else mp_pose.PoseLandmark.LEFT_ANKLE.value]
        vertical_diff = abs(ear[0] - ankle[0])
        if vertical_diff > 30:
            insights.append(
                f"[{name}] Possível inclinação anterior/posterior da cabeça/tronco (desvio de {vertical_diff:.1f}px).")

    elif name == "Costas":
        shoulder_tilt, s_diff = check_shoulder_tilt(landmarks)
        if shoulder_tilt:
            insights.append(f"[{name}] Assimetria nos ombros (diferença de altura de {s_diff:.1f}px).")

        pelvic_tilt, p_diff = check_pelvic_tilt(landmarks)
        if pelvic_tilt:
            insights.append(f"[{name}] Assimetria no quadril (diferença de altura de {p_diff:.1f}px).")

        scoliosis, scoliosis_diff = check_lateral_spine_curve(landmarks)
        if scoliosis:
            insights.append(f"[{name}] Indício de escoliose funcional (desvio médio de {scoliosis_diff:.1f}px).")

    if not insights:
        insights.append(f"[{name}] Nenhum desvio postural aparente encontrado.")

    return insights
