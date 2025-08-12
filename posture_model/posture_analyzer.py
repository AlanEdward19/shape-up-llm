import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees

mp_pose = mp.solutions.pose

class PostureAnalyzer:
    def __init__(self, static_image_mode=True, min_detection_confidence=0.5, min_visibility=0.5):
        self.pose = mp_pose.Pose(static_image_mode=static_image_mode,
                                 min_detection_confidence=min_detection_confidence)
        self.min_visibility = min_visibility

    @staticmethod
    def _lm_xyv(lm, w, h):
        return np.array([lm.x * w, lm.y * h, lm.visibility], dtype=np.float32)

    def get_landmarks(self, image_bgr):
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(image_rgb)
        if not res.pose_landmarks:
            return None, None
        pts = np.array([self._lm_xyv(lm, w, h) for lm in res.pose_landmarks.landmark], dtype=np.float32)
        return pts, (w, h)

    @staticmethod
    def _angle_deg(p1, p2):
        # ângulo da linha p1->p2 vs eixo horizontal
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        return degrees(atan2(dy, dx))

    @staticmethod
    def _rotate(points_xy, center_xy, angle_deg):
        ang = np.deg2rad(angle_deg)
        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang),  np.cos(ang)]], dtype=np.float32)
        return ((points_xy - center_xy) @ R.T) + center_xy

    def _shoulder_width(self, lms):
        L = mp_pose.PoseLandmark
        ls = lms[L.LEFT_SHOULDER.value][:2]
        rs = lms[L.RIGHT_SHOULDER.value][:2]
        return np.linalg.norm(ls - rs)

    def _height_proxy(self, lms):
        # Ombro médio até tornozelo médio (proxy de altura útil na imagem)
        L = mp_pose.PoseLandmark
        shoulder_mid = (lms[L.LEFT_SHOULDER.value][:2] + lms[L.RIGHT_SHOULDER.value][:2]) / 2
        ankle_mid = (lms[L.LEFT_ANKLE.value][:2] + lms[L.RIGHT_ANKLE.value][:2]) / 2
        return np.linalg.norm(shoulder_mid - ankle_mid)

    def _good_visibility(self, lms, idxs):
        return all(lms[i][2] >= self.min_visibility for i in idxs)

    def analyze(self, image_bgr, view_name: str):
        L = mp_pose.PoseLandmark
        lms, (w, h) = self.get_landmarks(image_bgr)
        if lms is None:
            return {"ok": False, "message": f"[{view_name}] Nenhum corpo detectado."}

        # métricas base
        sw = self._shoulder_width(lms)
        height_px = self._height_proxy(lms)

        if sw < 60 or height_px < 150:
            quality_note = "Distância/ enquadramento pode estar ruim (poucos pixels úteis)."
        else:
            quality_note = ""

        # Corrigir roll usando a linha dos ombros (horizontalizar)
        left_sh = lms[L.LEFT_SHOULDER.value][:2]
        right_sh = lms[L.RIGHT_SHOULDER.value][:2]
        cx = (left_sh + right_sh) / 2
        roll = self._angle_deg(right_sh, left_sh)  # se >0, ombro esquerdo mais alto
        pts2d = lms[:, :2]
        pts2d_rot = self._rotate(pts2d, cx, -roll)  # remove roll global
        lms_rot = lms.copy()
        lms_rot[:, :2] = pts2d_rot

        def pct_of_shoulder(d):
            return float(100.0 * d / max(sw, 1e-6))

        out = {
            "ok": True,
            "view": view_name,
            "quality_note": quality_note,
            "shoulder_width_px": float(sw),
            "height_proxy_px": float(height_px),
            "roll_correction_deg": float(roll),
            "metrics": {},
            "flags": [],
        }

        # --------- Frente/Costas (plano frontal) ----------
        if view_name in ["Front", "Back"]:
            need_idxs = [L.LEFT_SHOULDER.value, L.RIGHT_SHOULDER.value,
                         L.LEFT_HIP.value, L.RIGHT_HIP.value]
            if not self._good_visibility(lms, need_idxs):
                out["flags"].append("Baixa visibilidade de ombro/quadril.")
            # Ombros / Quadril tilt (em graus, após correção de roll)
            ls = lms_rot[L.LEFT_SHOULDER.value][:2]
            rs = lms_rot[L.RIGHT_SHOULDER.value][:2]
            lh = lms_rot[L.LEFT_HIP.value][:2]
            rh = lms_rot[L.RIGHT_HIP.value][:2]

            shoulder_tilt_deg = abs(self._angle_deg(rs, ls))
            pelvic_tilt_deg   = abs(self._angle_deg(rh, lh))

            # Midline desvio (ombros vs quadris)
            mid_sh = (ls + rs) / 2
            mid_hp = (lh + rh) / 2
            mid_diff_px = abs(mid_sh[0] - mid_hp[0])
            mid_diff_pct = pct_of_shoulder(mid_diff_px)

            out["metrics"].update({
                "shoulder_tilt_deg": float(shoulder_tilt_deg),
                "pelvic_tilt_deg": float(pelvic_tilt_deg),
                "midline_shift_px": float(mid_diff_px),
                "midline_shift_pct_shoulder": float(mid_diff_pct),
            })

            # Heurísticas de flag
            if shoulder_tilt_deg > 4:
                out["flags"].append(f"Tilt escapular suspeito (>4°): {shoulder_tilt_deg:.1f}°")
            if pelvic_tilt_deg > 4:
                out["flags"].append(f"Tilt pélvico suspeito (>4°): {pelvic_tilt_deg:.1f}°")
            if mid_diff_pct > 7:
                out["flags"].append(f"Desvio lateral do eixo (possível escoliose): {mid_diff_pct:.1f}% da largura dos ombros")

        # --------- Perfil (plano sagital) ----------
        if view_name in ["Right", "Left"]:
            left = (view_name == "Left")
            ear_i   = L.LEFT_EAR.value   if left else L.RIGHT_EAR.value
            sh_i    = L.LEFT_SHOULDER.value if left else L.RIGHT_SHOULDER.value
            hip_i   = L.LEFT_HIP.value   if left else L.RIGHT_HIP.value
            ankle_i = L.LEFT_ANKLE.value if left else L.RIGHT_ANKLE.value

            need_idxs = [ear_i, sh_i, hip_i, ankle_i]
            if not self._good_visibility(lms, need_idxs):
                out["flags"].append("Baixa visibilidade no perfil selecionado.")

            ear   = lms_rot[ear_i][:2]
            acrom = lms_rot[sh_i][:2]
            hip   = lms_rot[hip_i][:2]
            ankle = lms_rot[ankle_i][:2]

            # “Prumo”: diferença X entre orelha e maléolo
            prumo_head_trunk_px = abs(ear[0] - ankle[0])
            prumo_head_trunk_pct = pct_of_shoulder(prumo_head_trunk_px)

            # Forward head: distância horizontal orelha–acromion
            fhd_px  = abs(ear[0] - acrom[0])
            fhd_pct = pct_of_shoulder(fhd_px)

            # Alinhamento vertical: ângulo entre (orelha->tornozelo) e vertical
            dy = ankle[1] - ear[1]
            dx = ankle[0] - ear[0]
            align_deg = abs(90.0 - abs(degrees(atan2(dy, dx))))  # 0° = perfeitamente vertical

            out["metrics"].update({
                "sagittal_prumo_head_to_ankle_px": float(prumo_head_trunk_px),
                "sagittal_prumo_head_to_ankle_pct_shoulder": float(prumo_head_trunk_pct),
                "forward_head_ear_to_acromion_px": float(fhd_px),
                "forward_head_ear_to_acromion_pct_shoulder": float(fhd_pct),
                "head_trunk_alignment_deg": float(align_deg),
            })

            if fhd_pct > 15:
                out["flags"].append(f"Cabeça anteriorizada (>{15}%): {fhd_pct:.1f}% da largura dos ombros")
            if prumo_head_trunk_pct > 20:
                out["flags"].append(f"Desalinhamento cabeça–tornozelo: {prumo_head_trunk_pct:.1f}% da largura dos ombros")
            if align_deg > 5:
                out["flags"].append(f"Inclinação global do segmento cabeça–tornozelo: {align_deg:.1f}°")

        return out

    def draw_overlays(self, image_bgr, analysis):
        if not analysis.get("ok"):
            return image_bgr
        view = analysis["view"]
        L = mp_pose.PoseLandmark
        lms, _ = self.get_landmarks(image_bgr)
        if lms is None:
            return image_bgr

        # roll correction for drawing straight guides
        left_sh = lms[L.LEFT_SHOULDER.value][:2]
        right_sh = lms[L.RIGHT_SHOULDER.value][:2]
        cx = (left_sh + right_sh) / 2
        roll = self._angle_deg(right_sh, left_sh)
        pts2d = lms[:, :2]
        pts2d_rot = self._rotate(pts2d, cx, -roll)

        img = image_bgr.copy()

        def line(p, q, thickness=2):
            p = tuple(np.int32(p))
            q = tuple(np.int32(q))
            cv2.line(img, p, q, (0, 255, 0), thickness)

        if view in ["Front", "Back"]:
            ls = pts2d_rot[L.LEFT_SHOULDER.value]; rs = pts2d_rot[L.RIGHT_SHOULDER.value]
            lh = pts2d_rot[L.LEFT_HIP.value];      rh = pts2d_rot[L.RIGHT_HIP.value]
            mid_sh = (ls + rs) / 2; mid_hp = (lh + rh) / 2

            line(ls, rs, 3)   # ombros
            line(lh, rh, 3)   # quadris
            line(mid_sh, mid_hp, 1)  # midline

        if view in ["Right", "Left"]:
            left = (view == "Left")
            ear_i   = L.LEFT_EAR.value   if left else L.RIGHT_EAR.value
            sh_i    = L.LEFT_SHOULDER.value if left else L.RIGHT_SHOULDER.value
            hip_i   = L.LEFT_HIP.value   if left else L.RIGHT_HIP.value
            ankle_i = L.LEFT_ANKLE.value if left else L.RIGHT_ANKLE.value
            ear = pts2d_rot[ear_i]; sh = pts2d_rot[sh_i]; hip = pts2d_rot[hip_i]; an = pts2d_rot[ankle_i]

            line(ear, an, 1)  # prumo cabeça–tornozelo
            line(ear, sh, 2)  # orelha–acromion
            line(sh, hip, 1)  # ombro–quadril

        # Texto com flags
        y = 30
        for f in analysis.get("flags", [])[:5]:
            cv2.putText(img, f, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            y += 24
        return img
