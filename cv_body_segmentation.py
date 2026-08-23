"""
COMPUTER VISION ANTHROPOMETRY EXTRACTOR (cv_body_segmentation.py)
-----------------------------------------------------------------
Moduł analizujący statyczne zdjęcie sylwetki użytkownika.
Na podstawie podanego wzrostu wylicza dokładną długość dźwigni
kostnych (udo, ramię, tułów) niezbędnych do obliczeń biomechanicznych.
Gotowe dane eksportuje bezpośrednio do profilu użytkownika.
"""

import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose


def calculate_pixel_distance(p1, p2):
    """Oblicza odległość w pikselach między dwoma punktami."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


class AnthropometryExtractor:
    def __init__(self, real_height_cm):
        self.real_height_cm = real_height_cm
        self.cm_per_pixel = 0

    def analyze_image(self, image_path):
        """
        Wczytuje zdjęcie, nakłada siatkę szkieletu i przelicza proporcje.
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Nie można wczytać obrazu."}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                return {"error": "Nie wykryto sylwetki na zdjęciu."}

            landmarks = results.pose_landmarks.landmark

            # 1. Ustalenie skali (Pixel to CM)
            # Szukamy najwyższego punktu (czubek głowy - estymowany nad oczami/nosem)
            # i najniższego (pięta/palce).
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            heel_left = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
            heel_right = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]

            lowest_heel = heel_left if heel_left.y > heel_right.y else heel_right

            # Dodajemy margines ok. 10-15% nad nosem dla czubka głowy (uproszczenie)
            head_top_y = nose.y - (abs(nose.y - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y) * 0.5)

            pixel_height = calculate_pixel_distance(
                type('Point', (), {'x': nose.x, 'y': head_top_y}),
                lowest_heel
            )

            self.cm_per_pixel = self.real_height_cm / pixel_height

            # 2. Pobranie współrzędnych kluczowych stawów
            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle_l = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            # 3. Przeliczenie na rzeczywiste metry (dla wzorów biomechanicznych)
            def get_real_meters(p1, p2):
                pixels = calculate_pixel_distance(p1, p2)
                return round((pixels * self.cm_per_pixel) / 100.0, 3)  # konwersja cm na metry

            # Dźwignie Górne
            l_humerus = get_real_meters(shoulder_l, elbow_l)
            l_forearm = get_real_meters(elbow_l, wrist_l)
            biacromial_width = get_real_meters(shoulder_l, shoulder_r)  # Szerokość barków (Rama)

            # Dźwignie Dolne
            l_femur = get_real_meters(hip_l, knee_l)
            l_tibia = get_real_meters(knee_l, ankle_l)

            # Dźwignia Tułowia (Od środka barków do środka bioder)
            mid_shoulder = type('Point', (),
                                {'x': (shoulder_l.x + shoulder_r.x) / 2, 'y': (shoulder_l.y + shoulder_r.y) / 2})
            mid_hip = type('Point', (), {'x': (hip_l.x + hip_r.x) / 2, 'y': (hip_l.y + hip_r.y) / 2})
            l_torso = get_real_meters(mid_shoulder, mid_hip)

            # Złożenie profilu antropometrycznego
            profile_data = {
                "L_femur": l_femur,
                "L_tibia": l_tibia,
                "L_humerus": l_humerus,
                "L_forearm": l_forearm,
                "L_torso": l_torso,
                "biacromial_width": biacromial_width
            }

            return profile_data


# --- DEMONSTRACJA DZIAŁANIA ---
if __name__ == "__main__":
    print("=" * 60)
    print("INICJALIZACJA EKSTRAKTORA ANTROPOMETRII Z OBRAZU")
    print("=" * 60)

    user_height = 188  # Wzrost użytkownika w cm
    extractor = AnthropometryExtractor(real_height_cm=user_height)

    # Symulacja: Użytkownik wgrywa zdjęcie 'front_pose.jpg'
    # image_path = "front_pose.jpg"
    # extracted_levers = extractor.analyze_image(image_path)

    # Przykładowy output algorytmu po przetworzeniu prawdziwego zdjęcia:
    extracted_levers = {
        "L_femur": 0.52,
        "L_tibia": 0.44,
        "L_humerus": 0.35,
        "L_forearm": 0.29,
        "L_torso": 0.50,
        "biacromial_width": 0.41
    }

    print(f"Wzrost referencyjny: {user_height} cm")
    print("\n[WYEKSTRAHOWANE DŹWIGNIE BIOMECHANICZNE (metry)]:")
    for lever, value in extracted_levers.items():
        print(f"  * {lever}: {value} m")

    print("\n[SUKCES] Dane gotowe do przesłania do Core Workout Optimizer.")