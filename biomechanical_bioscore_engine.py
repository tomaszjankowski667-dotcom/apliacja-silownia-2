"""
BIOMECHANICAL BIO-SCORE ENGINE (biomechanical_bioscore_engine.py)
-----------------------------------------------------------------
Oblicza 3 Niezależne Filary Potencjału Biomechanicznego:
1. SYLWETKA (1-100): Realizacja potencjału LBM na podstawie ramy kostnej.
2. SIŁA (1-100): Rzeczywisty moment siły na dźwigniach vs Potencjał Maksymalny.
3. TECHNIKA (1-100): Kinematyka ruchu z wideo (ROM, Ekscentryka, Tor, Pauza).
Wypadkowa tworzy GLOBALNY BIO-SCORE użytkownika.
"""

import math
import numpy as np


# --- 1. FILAR: SYLWETKA (Physique Potential) ---

def calculate_physique_potential(height_cm, wrist_cm, ankle_cm, bf_percent, current_weight_kg):
    """
    Wylicza maksymalny naturalny potencjał beztłuszczowej masy ciała (LBM_max)
    na podstawie grubości kośćca (nadgarstek, kostka) i wzrostu (zmodyfikowany model Casey Butta).
    """
    height_inches = height_cm / 2.54
    wrist_inches = wrist_cm / 2.54
    ankle_inches = ankle_cm / 2.54

    # Maksymalna naturalna masa beztłuszczowa (LBM_max) przy 10% BF dla danego kośćca
    lbm_max_lbs = (height_inches ** 1.5) * (
            (math.sqrt(wrist_inches) / 22.667) + (math.sqrt(ankle_inches) / 21.893)
    ) * (18.0 / 224.0 + 1.0)

    lbm_max_kg = round(lbm_max_lbs * 0.45359237, 1)

    # Aktualna masa beztłuszczowa
    current_lbm = round(current_weight_kg * (1.0 - (bf_percent / 100.0)), 1)

    # Wynik bazy masy: % wypełnienia ramy kostnej
    lbm_ratio = current_lbm / lbm_max_kg
    base_physique_score = min(100, max(10, int(lbm_ratio * 100)))

    # Korekta za poziom zatłuszczenia (Body Fat Penalty / Bonus)
    # Optymalny zakres sylwetkowy: 10-13% dla mężczyzn. Powyżej 18% punkty spadają.
    if bf_percent <= 13.0:
        bf_modifier = 1.05  # Bonus za docięcie
    elif bf_percent <= 16.0:
        bf_modifier = 1.00
    elif bf_percent <= 20.0:
        bf_modifier = 0.90
    else:
        bf_modifier = max(0.65, 1.0 - ((bf_percent - 20.0) * 0.025))

    final_physique_score = min(100, max(1, int(base_physique_score * bf_modifier)))

    return {
        "score": final_physique_score,
        "current_lbm": current_lbm,
        "potential_max_lbm": lbm_max_kg,
        "potential_realized_pct": round((current_lbm / lbm_max_kg) * 100, 1),
        "bf_percent": bf_percent
    }


# --- 2. FILAR: SIŁA (Strength Potential vs Levers) ---

def calculate_strength_potential(user_levers, user_weights, current_lbm):
    """
    Ocenia siłę na podstawie momentów siły (Nm) generowanych na unikalnych
    dźwigniach kośćca w odniesieniu do maksymalnego potencjału neuromuskularnego.
    """
    G = 9.81

    # Teoretyczny maksymalny limit siłowy w Nm na kg LBM dla pełnej adaptacji
    THEORETICAL_MAX_TORQUE_PER_LBM = {
        "squat_pattern": 5.2,  # Nm / kg LBM na kolanie i biodrze
        "hinge_pattern": 4.8,  # Nm / kg LBM na prostownikach i taśmie tylnej
        "press_pattern": 4.2,  # Nm / kg LBM na stawie ramiennym
        "pull_pattern": 3.9  # Nm / kg LBM na najszerszym i zginaczach
    }

    # Rzeczywiste momenty obrotowe generowane przez użytkownika
    actual_torques = {
        "squat_pattern": (user_weights["squat"] * G * user_levers["L_femur"] * 0.75) / current_lbm,
        "hinge_pattern": (user_weights["deadlift_rdl"] * G * user_levers["L_torso"] * 0.65) / current_lbm,
        "press_pattern": (user_weights["bench_press"] * G * user_levers["L_humerus"] * 0.85) / current_lbm,
        "pull_pattern": (user_weights["pullup_pulldown"] * G * user_levers["L_humerus"] * 0.80) / current_lbm
    }

    exercise_scores = {}
    for key, actual_val in actual_torques.items():
        max_target = THEORETICAL_MAX_TORQUE_PER_LBM[key]
        realization_pct = (actual_val / max_target) * 100
        exercise_scores[key] = min(100, max(1, int(realization_pct)))

    avg_strength_score = int(np.mean(list(exercise_scores.values())))

    return {
        "score": avg_strength_score,
        "breakdown": exercise_scores,
        "details": "Wynik uwzględnia trudność długości Twoich kości (im dłuższa dźwignia, tym wyższy moment siły z danego ciężaru)."
    }


# --- 3. FILAR: TECHNIKA (Kinematic & Motor Quality) ---

def calculate_technique_score(video_metrics):
    """
    Ocenia jakość wykonania ruchu na podstawie zmiennych z modułu wideo:
    - rom_percentage (0 - 100%): wykorzystanie pełnego bezpiecznego zakresu
    - eccentric_control_score (1 - 100): kontrola fazy opuszczania i obecność pauzy w stretchu
    - path_stability_score (1 - 100): brak odchyleń trajektorii i brak kompensacji/bujania
    """
    rom_weight = 0.35
    eccentric_weight = 0.40
    stability_weight = 0.25

    weighted_score = (
            (video_metrics["rom_percentage"] * rom_weight) +
            (video_metrics["eccentric_control_score"] * eccentric_weight) +
            (video_metrics["path_stability_score"] * stability_weight)
    )

    final_technique_score = min(100, max(1, int(weighted_score)))

    return {
        "score": final_technique_score,
        "rom_score": video_metrics["rom_percentage"],
        "eccentric_stretch_score": video_metrics["eccentric_control_score"],
        "stability_score": video_metrics["path_stability_score"]
    }


# --- GŁÓWNY KALKULATOR BIO-SCORE ---

def evaluate_global_bio_score(user_data):
    # 1. Obliczenie Sylwetki
    physique_res = calculate_physique_potential(
        user_data["height_cm"],
        user_data["wrist_cm"],
        user_data["ankle_cm"],
        user_data["bf_percent"],
        user_data["weight_kg"]
    )

    # 2. Obliczenie Siły Biomechanicznej
    strength_res = calculate_strength_potential(
        user_data["levers"],
        user_data["weights"],
        physique_res["current_lbm"]
    )

    # 3. Obliczenie Techniki
    technique_res = calculate_technique_score(user_data["video_analysis"])

    # Globalny Bio-Score (Średnia ważona: Siła 40%, Sylwetka 35%, Technika 25%)
    global_score = int(
        (strength_res["score"] * 0.40) +
        (physique_res["score"] * 0.35) +
        (technique_res["score"] * 0.25)
    )

    # Tytuł Rangi RPG
    if global_score < 35:
        tier = "Nowicjusz Adaptacji"
    elif global_score < 60:
        tier = "Adept Hipertrofii"
    elif global_score < 80:
        tier = "Mistrz Biomechaniki"
    else:
        tier = "Genetyczny Tytan / Elita"

    return {
        "global_bio_score": global_score,
        "tier": tier,
        "pillars": {
            "sylwetka": physique_res,
            "siła": strength_res,
            "technika": technique_res
        }
    }


# --- DEMONSTRACJA I TEST DZIAŁANIA ---

if __name__ == "__main__":
    test_user_profile = {
        "name": "Kamil",
        "height_cm": 188,
        "weight_kg": 86.0,
        "bf_percent": 13.0,
        "wrist_cm": 18.0,  # Grubość nadgarstka (wyznacza ramę)
        "ankle_cm": 23.0,  # Grubość kostki
        "levers": {
            "L_femur": 0.53,  # Długie udo
            "L_torso": 0.50,  # Długi tułów
            "L_humerus": 0.36  # Długie ramię
        },
        "weights": {
            "squat": 130,
            "deadlift_rdl": 145,
            "bench_press": 95,
            "pullup_pulldown": 85
        },
        "video_analysis": {
            "rom_percentage": 95,  # Prawie pełny zakres
            "eccentric_control_score": 88,  # Dobra 3-sekundowa ekscentryka i pauza
            "path_stability_score": 90  # Brak kołysania tułowiem
        }
    }

    results = evaluate_global_bio_score(test_user_profile)

    print("=" * 65)
    print(f"RAPORT BIO-SCORE DLA: {test_user_profile['name']}")
    print(f"GLOBALNY BIO-SCORE: {results['global_bio_score']} / 100  |  Ranga: {results['tier']}")
    print("=" * 65)

    print("\n[1] FILAR: SYLWETKA (1-100):", results['pillars']['sylwetka']['score'], "/ 100")
    print(f"  * Aktualna masa mięśniowa (LBM): {results['pillars']['sylwetka']['current_lbm']} kg")
    print(f"  * Maksymalny naturalny potencjał LBM: {results['pillars']['sylwetka']['potential_max_lbm']} kg")
    print(f"  * Wykorzystanie genetycznej ramy: {results['pillars']['sylwetka']['potential_realized_pct']}%")
    print(f"  * Poziom tkanki tłuszczowej (BF): {results['pillars']['sylwetka']['bf_percent']}%")

    print("\n[2] FILAR: SIŁA BIOMECHANICZNA (1-100):", results['pillars']['siła']['score'], "/ 100")
    for pattern, score in results['pillars']['siła']['breakdown'].items():
        print(f"  * {pattern:18}: {score:3d} / 100")

    print("\n[3] FILAR: TECHNIKA & KINEMATYKA (1-100):", results['pillars']['technika']['score'], "/ 100")
    print(f"  * Wykorzystanie zakresu (ROM): {results['pillars']['technika']['rom_score']}%")
    print(
        f"  * Kontrola ekscentryki i rozciągnięcia: {results['pillars']['technika']['eccentric_stretch_score']} / 100")
    print(f"  * Stabilność toru ruchu (Bar Path): {results['pillars']['technika']['stability_score']} / 100")
    print("=" * 65)