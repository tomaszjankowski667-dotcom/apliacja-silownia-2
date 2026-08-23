"""
MAIN BACKEND API (main_api.py)
------------------------------
Serwer spinający ręczne dane użytkownika z AI oraz biomechaniką.
Przyjmuje żądania z aplikacji mobilnej (wzrost, waga, zdjęcia),
uruchamia analizę i zwraca gotowy Bio-Score oraz plan treningowy.
"""

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import json
import uvicorn

# Symulacja importów naszych poprzednich modułów
# from cv_body_segmentation import AnthropometryExtractor
# from core_workout_optimizer import calculate_us_navy_bf, analyze_levers, calculate_bio_score, generate_custom_split
# from dynamic_auto_programmer import WorkoutProgrammer

app = FastAPI(title="Biomechanics & Bio-Score API", version="1.0")


# --- 1. MODELE DANYCH (Ręczne wejście od użytkownika) ---

class UserBasicInfo(BaseModel):
    user_id: int
    name: str
    gender: str
    height_cm: float  # Ręczny, krytyczny punkt odniesienia dla AI
    weight_kg: float  # Niezbędne do wyliczenia LBM i momentów siły
    waist_cm: float
    neck_cm: float


class WorkoutPerformance(BaseModel):
    squat_kg: float
    bench_kg: float
    deadlift_kg: float
    pullup_kg: float


# --- 2. ENDPOINTY (KOMUNIKACJA Z APLIKACJĄ) ---

@app.post("/analyze-physique/")
async def analyze_physique_endpoint(
        height_cm: float = Form(...),
        weight_kg: float = Form(...),
        photo: UploadFile = File(...)
):
    """
    KROK 1: Użytkownik podaje w aplikacji swój wzrost i wagę, a następnie robi zdjęcie.
    Endpoint przelicza piksele ze zdjęcia na rzeczywiste metry dzięki zmiennej height_cm.
    """
    # 1. Zapisanie przesłanego zdjęcia na serwerze (tymczasowo)
    file_location = f"temp_{photo.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(photo.file.read())

    # 2. Uruchomienie ekstrakcji AI (wymaga ręcznego wzrostu do kalibracji skali!)
    # extractor = AnthropometryExtractor(real_height_cm=height_cm)
    # levers = extractor.analyze_image(file_location)

    # Symulacja wyniku dla demonstracji:
    levers = {
        "L_femur": 0.53, "L_tibia": 0.44,
        "L_torso": 0.50, "L_humerus": 0.36
    }

    return {
        "status": "success",
        "message": "Kalibracja AI powiodła się na podstawie podanego wzrostu.",
        "extracted_levers": levers
    }


@app.post("/generate-profile/")
async def generate_full_profile(user: UserBasicInfo, performance: WorkoutPerformance, levers: dict):
    """
    KROK 2: Połączenie ręcznych pomiarów, ciężarów roboczych i dźwigni z AI
    w celu wyliczenia Bio-Score i wygenerowania planu.
    """
    # 1. Wyliczenie prawdziwej bazy mięśniowej (LBM)
    # bf_percent = calculate_us_navy_bf(user.gender, user.height_cm, user.waist_cm, user.neck_cm)
    bf_percent = 13.5  # Symulacja
    lbm_kg = round(user.weight_kg * (1.0 - (bf_percent / 100.0)), 1)

    # 2. Analiza Dźwigni i Bio-Score
    # lever_analysis = analyze_levers(user_dict)
    # bio_score = calculate_bio_score(user_dict, lbm_kg)
    bio_score = {"global_score": 78, "rank": "Weteran (Zaawansowana Przebudowa)"}  # Symulacja

    # 3. Dynamiczny Generator Planu
    # programmer = WorkoutProgrammer(user_dict, "PROGRESJA W NORMIE")
    # workout_plan = programmer.generate_workout("Legs")

    workout_plan = {
        "day": "Legs",
        "routine": [
            {"muscle": "Czworogłowe", "exercise": "Hack Przysiad na Maszynie", "sets": 4},
            {"muscle": "Tył Uda", "exercise": "Rumuński Martwy Ciąg", "sets": 3}
        ]
    }

    return {
        "user_name": user.name,
        "body_composition": {
            "weight_input": user.weight_kg,
            "bf_estimated": bf_percent,
            "lean_body_mass": lbm_kg
        },
        "bio_score": bio_score,
        "prescribed_workout": workout_plan
    }


@app.get("/health")
def health_check():
    return {"status": "Serwer biomechaniczny działa z pełną mocą."}


if __name__ == "__main__":
    print("=" * 50)
    print("STARTOWANIE SERWERA API BIOMECHANIKI")
    print("Nasłuchiwanie na porcie 8000...")
    print("=" * 50)
    # uvicorn.run(app, host="0.0.0.0", port=8000)