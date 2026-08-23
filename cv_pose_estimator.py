"""
COMPUTER VISION POSE & BAR PATH ESTIMATOR (cv_pose_estimator.py)
----------------------------------------------------------------
Nowoczesny silnik analizy techniki:
1. Odrzuca mimikę twarzy (zostawia tylko pozycję głowy).
2. Śledzi trajektorię sztangi (Barbell Path Tracking) w czasie rzeczywistym.
3. Wykrywa fazy ruchu, powtórzenia i kąty stawowe.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
from collections import deque

def calculate_angle(a, b, c):
    """Oblicza kąt między 3 punktami w stopniach."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# Połączenia szkieletu BEZ twarzy (punkty 11-32)
BODY_CONNECTIONS = [
    (11, 12),           # Barki (Obręcz barkowa)
    (11, 13), (13, 15), # Lewe ramię: Bark -> Łokieć -> Nadgarstek
    (12, 14), (14, 16), # Prawe ramię: Bark -> Łokieć -> Nadgarstek
    (11, 23), (12, 24), # Tułów: Barki -> Biodra
    (23, 24),           # Pas biodrowy
    (23, 25), (25, 27), # Lewa noga: Biodro -> Kolano -> Kostka
    (24, 26), (26, 28), # Prawa noga: Biodro -> Kolano -> Kostka
    (27, 29), (28, 30), # Stopy
    (29, 31), (30, 32)
]

def run_vision_analysis(video_source=0):
    model_path = "pose_landmarker.task"
    if not os.path.exists(model_path):
        print(f"BŁĄD: Brak pliku '{model_path}' w głównym folderze projektu!")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_source)

    # Bufor trajektorii sztangi (ostatnie 40 klatek toru ruchu)
    bar_path_history = deque(maxlen=40)

    rep_count = 0
    current_phase = "eccentric"
    eccentric_start_time = 0

    print("Uruchomiono analizę biomechaniczną. Wciśnij 'q', aby zakończyć.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(time.time() * 1000)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]

            # 1. PUNKTY STAWOWE DO BIOMECHANIKI
            # Prawa strona
            r_shoulder = [landmarks[12].x, landmarks[12].y]
            r_elbow = [landmarks[14].x, landmarks[14].y]
            r_wrist = [landmarks[16].x, landmarks[16].y]
            r_hip = [landmarks[24].x, landmarks[24].y]
            r_knee = [landmarks[26].x, landmarks[26].y]
            r_ankle = [landmarks[28].x, landmarks[28].y]

            # Lewy nadgarstek do wyznaczenia środka sztangi
            l_wrist = [landmarks[15].x, landmarks[15].y]

            # Pozycja sztangi = środek między nadgarstkami (lub prawy nadgarstek z profilu)
            bar_x = int(((r_wrist[0] + l_wrist[0]) / 2.0) * w)
            bar_y = int(((r_wrist[1] + l_wrist[1]) / 2.0) * h)
            bar_path_history.append((bar_x, bar_y))

            # 2. OBLICZENIA KĄTÓW
            arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            leg_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # Kąt garbienia się: Głowa (nos: 0) -> Bark (12) -> Biodro (24)
            head = [landmarks[0].x, landmarks[0].y]
            posture_angle = calculate_angle(head, r_shoulder, r_hip)

            # Logika powtórzeń (na przykładzie zgięcia łokcia / przysiadu)
            active_angle = arm_angle if arm_angle < 150 else leg_angle
            curr_sec = time.time()

            if active_angle > 150:
                if current_phase == "concentric":
                    rep_count += 1
                    current_phase = "eccentric"
                    eccentric_start_time = curr_sec
            elif active_angle < 90:
                if current_phase == "eccentric":
                    current_phase = "concentric"

            # 3. RYSOWANIE TORU SZTANGI (BAR PATH)
            for i in range(1, len(bar_path_history)):
                if bar_path_history[i - 1] is None or bar_path_history[i] is None:
                    continue
                # Zanikająca grubość linii (efekt ogona świetlnego)
                thickness = int(np.sqrt(40 / float(i + 1)) * 2.5)
                # Kolor neonowo-błękitny dla toru sztangi
                cv2.line(frame, bar_path_history[i - 1], bar_path_history[i], (255, 255, 0), thickness)

            # 4. RYSOWANIE SZKIELETU CIAŁA (BEZ TWARZY)
            for start_idx, end_idx in BODY_CONNECTIONS:
                pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Rysujemy punkty stawowe tylko dla ciała (indeksy >= 11)
            for idx, lm in enumerate(landmarks):
                if idx >= 11:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Punkt centralny sztangi
            cv2.circle(frame, (bar_x, bar_y), 7, (0, 255, 255), -1)

            # 5. DASHBOARD INFORMACYJNY
            cv2.putText(frame, f"Reps: {rep_count}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Phase: {current_phase.upper()}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Angle: {int(active_angle)}*", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Wskaźnik postawy (czy się nie garbi)
            posture_status = "OK" if posture_angle > 140 else "GARBIENIE!"
            posture_color = (0, 255, 0) if posture_status == "OK" else (0, 0, 255)
            cv2.putText(frame, f"Postawa: {posture_status}", (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)

        cv2.imshow("Biomechanics AI Engine", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    run_vision_analysis(0)