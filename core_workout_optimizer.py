"""
CORE WORKOUT OPTIMIZER & BIO-SCORE ENGINE (core_workout_optimizer.py)
---------------------------------------------------------------------
Centralny silnik analityczny:
1. Estymuje LBM (Lean Body Mass) i %BF metodą US Navy.
2. Analizuje dźwignie biomechaniczne użytkownika.
3. Wylicza sprawiedliwy Bio-Score (1-100) per partia i globalnie.
4. Generuje spersonalizowany plan treningowy (Auto-Programmer).
5. Zapisuje profil do formatu JSON.
"""

import json
import math
import numpy as np


# --- 1. FUNKCJE SOMATYCZNE I DŹWIGNIOWE ---

def calculate_us_navy_bf(gender, height_cm, waist_cm, neck_cm, hip_cm=None):
    """
    Szacuje procent tkanki tłuszczowej wzorem US Navy.
    """
    if gender.lower() == "male":
        # Wzór dla mężczyzn
        bf_percent = 495.0 / (
                    1.0324 - 0.19077 * math.log10(waist_cm - neck_cm) + 0.15456 * math.log10(height_cm)) - 450.0
    else:
        # Wzór dla kobiet
        if hip_cm is None:
            hip_cm = waist_cm * 1.15
        bf_percent = 495.0 / (1.29579 - 0.35004 * math.log10(waist_cm + hip_cm - neck_cm) + 0.22100 * math.log10(
            height_cm)) - 450.0

    return max(4.0, min(50.0, round(bf_percent, 1)))


def analyze_levers(user):
    """
    Identyfikuje specyfikę budowy szkieletu.
    """
    height_m = user["height_cm"] / 100.0
    femur_ratio = user["L_femur"] / height_m
    torso_ratio = user["L_torso"] / height_m
    arm_ratio = (user["L_humerus"] + user["L_forearm"]) / height_m

    profile_tags = []

    # Ocena dźwigni przysiadu / dołu ciała
    if femur_ratio > 0.27 and torso_ratio < 0.27:
        profile_tags.append("DŁUGIE DŹWIGNIE UD (Niekorzystne pod klasyczny przysiad ze sztangą)")
    elif femur_ratio < 0.25:
        profile_tags.append("KRÓTKIE DŹWIGNIE UD (Dominacja czwórogłowych w przysiadzie)")

    # Ocena dźwigni wyciskania / góry ciała
    if arm_ratio > 0.36:
        profile_tags.append("DŁUGIE RAMIONA (Dłuższy tor wyciskania, przewaga w ciągach)")
    elif arm_ratio < 0.33:
        profile_tags.append("KRÓTKIE RAMIONA (Krótki tor wyciskania, przewaga w wyciskaniu)")

    return {
        "femur_ratio": round(femur_ratio, 3),
        "torso_ratio": round(torso_ratio, 3),
        "arm_ratio": round(arm_ratio, 3),
        "tags": profile_tags
    }


# --- 2. SILNIK BIO-SCORE (1-100) ---

# Standardy bazowe momentu siły w odniesieniu do LBM (Nm/kg LBM) dla poziomu elitarnego (100 pkt)
ELITE_TORQUE_STANDARDS = {
    "quads": 4.8,  # Np. Hack Squat / Leg Press przeliczony na moment
    "hamstrings": 2.6,  # RDL / Leg Curl
    "chest": 3.8,  # Wyciskania ze sztangą/hantlami
    "lats": 3.5,  # Ściągania drążka / podciąganie
    "upper_back": 3.2,  # Wiosłowania z podparciem klatki
    "front_delts": 2.2,  # Wyciskania nad głowę
    "side_delts": 1.1,  # Wznosy bokiem
    "biceps": 1.4,  # Uginania ramion
    "triceps": 2.0  # Wyprosty ramion
}


def calculate_bio_score(user_data, lbm_kg):
    """
    Kalkuluje poziom zaawansowania (1-100) per partia i globalnie.
    Uwzględnia unikalne ramię siły, a nie tylko suchy ciężar.
    """
    scores = {}
    weights = user_data["weights"]

    # 1. Czworogłowe (Quads): Ciężar w przysiadzie/hack x ramię kości udowej
    quad_torque = (weights.get("squat_pattern", 100) * 9.81 * user_data["L_femur"] * 0.7) / lbm_kg
    scores["Czworogłowe"] = min(100, max(1, int((quad_torque / ELITE_TORQUE_STANDARDS["quads"]) * 100)))

    # 2. Tył uda (Hamstrings): Ciężar RDL x ramię tułowia i biodra
    ham_torque = (weights.get("hinge_pattern", 110) * 9.81 * user_data["L_torso"] * 0.6) / lbm_kg
    scores["Kulszowo-Goleniowe"] = min(100, max(1, int((ham_torque / ELITE_TORQUE_STANDARDS["hamstrings"]) * 100)))

    # 3. Klatka (Chest): Ciężar wyciskania x ramię kości ramiennej
    chest_torque = (weights.get("bench_pattern", 80) * 9.81 * user_data["L_humerus"] * 0.8) / lbm_kg
    scores["Klatka Piersiowa"] = min(100, max(1, int((chest_torque / ELITE_TORQUE_STANDARDS["chest"]) * 100)))

    # 4. Najszerszy Grzbietu (Lats): Ciężar ściągania x ramię ramienia
    lats_torque = (weights.get("pull_pattern", 75) * 9.81 * user_data["L_humerus"] * 0.85) / lbm_kg
    scores["Najszerszy Grzbietu"] = min(100, max(1, int((lats_torque / ELITE_TORQUE_STANDARDS["lats"]) * 100)))

    # 5. Górne Plecy (Upper Back)
    ub_torque = (weights.get("row_pattern", 70) * 9.81 * user_data["L_humerus"] * 0.8) / lbm_kg
    scores["Górne Plecy"] = min(100, max(1, int((ub_torque / ELITE_TORQUE_STANDARDS["upper_back"]) * 100)))

    # 6. Boczny Akton Barku (Side Delts)
    lat_torque = (weights.get("lateral_raise", 14) * 9.81 * (user_data["L_humerus"] + user_data["L_forearm"])) / lbm_kg
    scores["Boczny Akton Barku"] = min(100, max(1, int((lat_torque / ELITE_TORQUE_STANDARDS["side_delts"]) * 100)))

    # 7. Ramiona (Biceps + Triceps)
    bic_torque = (weights.get("biceps_curl", 35) * 9.81 * user_data["L_forearm"]) / lbm_kg
    scores["Biceps"] = min(100, max(1, int((bic_torque / ELITE_TORQUE_STANDARDS["biceps"]) * 100)))

    tri_torque = (weights.get("triceps_ext", 35) * 9.81 * user_data["L_forearm"]) / lbm_kg
    scores["Triceps"] = min(100, max(1, int((tri_torque / ELITE_TORQUE_STANDARDS["triceps"]) * 100)))

    # Średni Globalny Bio-Score
    global_score = int(np.mean(list(scores.values())))

    # Ranga RPG na podstawie wyniku
    if global_score < 30:
        rank = "Nowicjusz (Adaptacja Neurologiczna)"
    elif global_score < 65:
        rank = "Adept (Hipertrofia Strukturalna)"
    elif global_score < 85:
        rank = "Weteran (Zaawansowana Przebudowa)"
    else:
        rank = "Tytan (Elita Biomechaniczna)"

    return {
        "global_score": global_score,
        "rank": rank,
        "breakdown": scores
    }


# --- 3. AUTO-PROGRAMMER (DOBÓR ĆWICZEŃ POD DŹWIGNIE) ---

def generate_custom_split(user, levers):
    """
    Tworzy zoptymalizowany zestaw ćwiczeń, filtrując słabe biomechanicznie wzorce.
    """
    is_long_femur = levers["femur_ratio"] > 0.26
    is_long_torso = levers["torso_ratio"] > 0.27

    # Push Day
    push_exercises = [
        "Wyciskanie Hantli / Maszyny na Skosie Dodatnim (30-45°)",
        "Dipsy na Poręczach lub Maszyna Hammer Press",
        "Wznosy Bokiem w Leżeniu na Ławce Skośnej (Incline Side Raise)",
        "Wyciąg Nad Głową na Triceps (Overhead Cable Extension)"
    ]

    # Pull Day (Korekta o długi tułów - eliminacja martwego ciągu/wiosła w opadzie na rzecz podparcia)
    if is_long_torso:
        pull_exercises = [
            "Ściąganie Wyciągu Jednorącz (Optymalne rozciągnięcie latsów)",
            "Wiosłowanie na Maszynie z Oparciem Klatki (Chest-Supported Row)",
            "Odwrotne Rozpiętki na Maszynie (Reverse Pec-Deck)",
            "Uginanie Ramion na Modlitewniku ze Sztangą Łamaną"
        ]
    else:
        pull_exercises = [
            "Podciąganie z Ciężarem / Ściąganie Drążka",
            "Wiosłowanie Półsztangą (T-Bar Row)",
            "Odwrotne Rozpiętki z Linkami Wyciągu Górnego",
            "Uginanie ze Sztangą Stojąc"
        ]

    # Legs Day (Korekta o długie kości udowe - eliminacja przysiadu ze sztangą na karku)
    if is_long_femur:
        leg_exercises = [
            "Hack Przysiad na Maszynie / Pasmowy (Głębokie zgięcie kolana)",
            "Uginanie Nóg Leżąc na Maszynie (Lying Leg Curl)",
            "Wypychanie Ciężaru na Suwnicy (Leg Press - stopy nisko)",
            "Rumuński Martwy Ciąg z Hantlami (RDL - skupienie na biodrze)",
            "Wspięcia na Palce na Suwnicy (Leg Press Calf Raises)"
        ]
    else:
        leg_exercises = [
            "Przysiad ze Sztangą na Plecach (Back Squat)",
            "Rumuński Martwy Ciąg ze Sztangą (RDL)",
            "Prostowanie Nóg na Maszynie (Leg Extension)",
            "Uginanie Nóg Siedząc na Maszynie",
            "Ośle Wspięcia (Donkey Calf Raises)"
        ]

    return {
        "Push": push_exercises,
        "Pull": pull_exercises,
        "Legs": leg_exercises
    }


# --- 4. GŁÓWNA PROCEDURA WYKONAWCZA ---

def run_user_analysis(user_profile):
    print("=" * 65)
    print(f"ANALIZA BIOMECHANICZNA I BIO-SCORE: {user_profile['name']}")
    print("=" * 65)

    # 1. Obliczenie składu ciała
    bf = calculate_us_navy_bf(
        user_profile["gender"],
        user_profile["height_cm"],
        user_profile["waist_cm"],
        user_profile["neck_cm"]
    )
    lbm = round(user_profile["weight_kg"] * (1.0 - (bf / 100.0)), 1)

    print(f"\n[1] SKŁAD CIAŁA:")
    print(f"  * Waga całkowita: {user_profile['weight_kg']} kg")
    print(f"  * Tkanka tłuszczowa (US Navy): {bf}%")
    print(f"  * Beztłuszczowa Masa Ciała (LBM): {lbm} kg")

    # 2. Analiza dźwigni
    levers = analyze_levers(user_profile)
    print(f"\n[2] DŹWIGNIE SZKIELETOWE:")
    for tag in levers["tags"]:
        print(f"  * [ALERT]: {tag}")

    # 3. Bio-Score 1-100
    bio_data = calculate_bio_score(user_profile, lbm)
    print(f"\n[3] WYNIK BIO-SCORE (GAMIFIKACJA):")
    print(f"  * GLOBALNY BIO-SCORE: {bio_data['global_score']} / 100")
    print(f"  * RANGA POSTACI: {bio_data['rank']}")
    print("  * Wyniki cząstkowe partii:")
    for part, score in bio_data["breakdown"].items():
        print(f"    - {part:22}: {score:3d} / 100")

    # 4. Spersonalizowany Plan (Auto-Programmer)
    split = generate_custom_split(user_profile, levers)
    print(f"\n[4] ZOPTYMALIZOWANY PLAN TRENINGOWY (PUSH / PULL / LEGS):")
    for day, exs in split.items():
        print(f"\n  [{day.upper()} DAY]:")
        for i, ex in enumerate(exs, 1):
            print(f"    {i}. {ex}")

    # 5. Eksport JSON
    export_payload = {
        "user_name": user_profile["name"],
        "metrics": {"weight_kg": user_profile["weight_kg"], "bf_percent": bf, "lbm_kg": lbm},
        "bio_score": bio_data,
        "levers": levers,
        "prescribed_split": split
    }

    with open("user_bio_profile.json", "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("Zapisano pełny profil użytkownika do 'user_bio_profile.json'.")
    print("=" * 65)


# --- PRZYKŁADOWY PROFIL TESTOWY (Użytkownik o długich dźwigniach) ---
if __name__ == "__main__":
    test_user = {
        "name": "Kamil (Twój Profil - Długie Kości)",
        "gender": "male",
        "height_cm": 188,
        "weight_kg": 86.0,
        "waist_cm": 83.0,
        "neck_cm": 39.0,
        "L_femur": 0.53,  # Długie udo
        "L_torso": 0.50,  # Długi tułów
        "L_humerus": 0.36,  # Długie ramię
        "L_forearm": 0.30,  # Długie przedramię
        "weights": {
            "squat_pattern": 110,
            "hinge_pattern": 130,
            "bench_pattern": 85,
            "pull_pattern": 75,
            "row_pattern": 70,
            "lateral_raise": 14,
            "biceps_curl": 35,
            "triceps_ext": 30
        }
    }

    run_user_analysis(test_user)