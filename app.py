import os
import urllib.request
import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BiomechanicsEngine:
    def __init__(self):
        # Automatyczne pobranie oficjalnego modelu AI
        self.model_path = "pose_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Pobieram oficjalny model AI MediaPipe (jednorazowo)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Model pobrany pomyślnie!")

        # Inicjalizacja silnika PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def analyze(self, image_path: str, injuries: list, mobility: dict):
        if not os.path.exists(image_path):
            return {
                "error": f"Nie znaleziono pliku '{image_path}'. Wklej zdjęcie i nazwij je test.jpg w folderze projektu."}

        # Wczytanie obrazu
        mp_image = mp.Image.create_from_file(image_path)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            return {"error": "AI nie wykryło postaci na zdjęciu."}

        # Pobranie punktów stawów (11=Bark, 23=Biodro, 25=Kolano)
        landmarks = detection_result.pose_landmarks[0]
        shoulder = landmarks[11]
        hip = landmarks[23]
        knee = landmarks[25]

        # Matematyka wektorowa
        torso = np.sqrt((shoulder.x - hip.x) ** 2 + (shoulder.y - hip.y) ** 2)
        femur = np.sqrt((hip.x - knee.x) ** 2 + (hip.y - knee.y) ** 2)

        ratio = round(femur / torso, 2) if torso > 0 else 0

        # Logika biomechaniczna
        score_board = {}
        back_squat_score = 10
        notes = []

        if ratio > 0.85:
            back_squat_score -= 4
            notes.append("Długa kość udowa generuje duży moment zginający na dolne plecy.")

        if "bol_plecow" in injuries:
            back_squat_score -= 5
            notes.append("Kwarantanna: Wykryto ból pleców – drastyczna redukcja nacisku osiowego.")

        if mobility.get("kostka") == "slaba":
            notes.append("Słaba mobilność kostki: Wymagane podkładki pod pięty.")

        score_board["Back Squat"] = max(0, back_squat_score)

        hack_squat_score = 8
        if ratio > 0.85:
            hack_squat_score += 2
        score_board["Hack Squat / Suwnica"] = hack_squat_score

        return {
            "biometrics": {
                "torso_length": round(torso, 4),
                "femur_length": round(femur, 4),
                "femur_torso_ratio": ratio,
            },
            "exercise_scores": score_board,
            "notes": notes
        }

    def close(self):
        """Bezpieczne zwolnienie pamięci silnika AI."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


if __name__ == "__main__":
    engine = BiomechanicsEngine()
    try:
        wynik = engine.analyze(
            image_path="test.jpg",
            injuries=["bol_plecow"],
            mobility={"kostka": "slaba"}
        )
        print(json.dumps(wynik, indent=2, ensure_ascii=False))
    finally:
        engine.close()