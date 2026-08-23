"""
LEGS MODULE (legs.py)
---------------------
Kompletny atlas ćwiczeń dla kończyn dolnych i miednicy.
Podział na 5 grup i precyzyjne aktony:
1. CZWOROGŁOWE UDA (czworoglowe_prosty_uda, czworoglowe_obszerne)
2. KULSZOWO-GOLENIOWE (dwuglowe_glowa_dluga, dwuglowe_glowa_krotka)
3. POŚLADKI (posladkowy_wielki, posladkowy_sredni_maly)
4. ŁYDKI (lydka_brzuchaty, lydka_plaszczkowaty)
5. PRZYWODZICIELE (przywodziciel_wielki)
"""

def analyze_legs_exercises(biometrics: dict, injuries: list) -> dict:
    femur_torso_ratio = biometrics.get("femur_torso_ratio", 0.80)
    tibia_femur_ratio = biometrics.get("tibia_femur_ratio", 0.85)

    legs_database = {
        # =========================================================================
        # 1. MIĘŚNIE CZWOROGŁOWE UDA (QUADS)
        # =========================================================================
        "Przysiad_Sztanga_High_Bar": {
            "name": "Przysiad ze Sztangą na Karku (High Bar)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "HIGH",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.7,
                "czworoglowe_prosty_uda": 0.3,
                "posladkowy_wielki": 0.5,
                "przywodziciel_wielki": 0.4,
                "prostowniki_grzbietu": 0.4
            }
        },
        "Przysiad_Sztanga_Low_Bar": {
            "name": "Przysiad ze Sztangą na Karku (Low Bar)",
            "primary_target": "posladkowy_wielki",
            "spinal_axial_load": "HIGH",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "posladkowy_wielki": 0.6,
                "czworoglowe_obszerne": 0.5,
                "przywodziciel_wielki": 0.5,
                "prostowniki_grzbietu": 0.6
            }
        },
        "Przysiad_Front_Squat": {
            "name": "Przysiad ze Sztangą z Przodu (Front Squat)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "HIGH",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.75,
                "czworoglowe_prosty_uda": 0.25,
                "posladkowy_wielki": 0.3,
                "prostowniki_grzbietu": 0.5
            }
        },
        "Suwnica_Leg_Press": {
            "name": "Wyciskanie Nóg na Suwnicy (Leg Press)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "LOW",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.75,
                "czworoglowe_prosty_uda": 0.25,
                "posladkowy_wielki": 0.4,
                "przywodziciel_wielki": 0.3
            }
        },
        "Hack_Squat_Maszyna": {
            "name": "Hack Przysiady na Maszynie (Hack Squat)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "LOW",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.8,
                "czworoglowe_prosty_uda": 0.2,
                "posladkowy_wielki": 0.3,
                "przywodziciel_wielki": 0.3
            }
        },
        "Pendulum_Squat": {
            "name": "Przysiady na Maszynie Wahadłowej (Pendulum Squat)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "LOW",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 10.0,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.85,
                "czworoglowe_prosty_uda": 0.15,
                "posladkowy_wielki": 0.3
            }
        },
        "Przysiad_Bulgarski": {
            "name": "Przysiad Bułgarski (Bulgarian Split Squat)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "LOW",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.6,
                "posladkowy_wielki": 0.5,
                "posladkowy_sredni_maly": 0.3,
                "przywodziciel_wielki": 0.3
            }
        },
        "Prostowanie_Nog_Siedzac": {
            "name": "Prostowanie Nóg na Maszynie Siedząc (Leg Extension)",
            "primary_target": "czworoglowe_prosty_uda",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "HIGH",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "czworoglowe_prosty_uda": 0.65,
                "czworoglowe_obszerne": 0.35
            }
        },
        "Wykroki_Zakroki": {
            "name": "Wykroki i Zakroki (Hantle / Sztanga / Smith)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "MEDIUM",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.6,
                "posladkowy_wielki": 0.5,
                "posladkowy_sredni_maly": 0.3
            }
        },
        "Przysiad_Smith": {
            "name": "Przysiad na Maszynie Smitha",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "MEDIUM",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.75,
                "posladkowy_wielki": 0.4,
                "przywodziciel_wielki": 0.3
            }
        },
        "Sissy_Squat": {
            "name": "Sissy Squat (Przysiad Sissy)",
            "primary_target": "czworoglowe_prosty_uda",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "HIGH", # Bardzo wysokie siły na aparat rzepkowy
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "czworoglowe_prosty_uda": 0.8,
                "czworoglowe_obszerne": 0.2
            }
        },
        "Belt_Squat": {
            "name": "Przysiad na Maszynie Belt Squat (z pasem)",
            "primary_target": "czworoglowe_obszerne",
            "spinal_axial_load": "ZERO", # Zerowe obciążenie kręgosłupa!
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "czworoglowe_obszerne": 0.8,
                "posladkowy_wielki": 0.4,
                "przywodziciel_wielki": 0.3
            }
        },

        # =========================================================================
        # 2. MIĘŚNIE KULSZOWO-GOLENIOWE (HAMSTRINGS)
        # =========================================================================
        "RDL_Rumunski_Martwy_Ciag": {
            "name": "Rumuński Martwy Ciąg (RDL)",
            "primary_target": "dwuglowe_glowa_dluga",
            "spinal_axial_load": "MEDIUM",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "dwuglowe_glowa_dluga": 0.7,
                "posladkowy_wielki": 0.6,
                "przywodziciel_wielki": 0.4,
                "prostowniki_grzbietu": 0.4
            }
        },
        "Martwy_Ciag_Proste_Nogi": {
            "name": "Martwy Ciąg na Prostych Nogach",
            "primary_target": "dwuglowe_glowa_dluga",
            "spinal_axial_load": "HIGH",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "dwuglowe_glowa_dluga": 0.8,
                "posladkowy_wielki": 0.5,
                "prostowniki_grzbietu": 0.6
            }
        },
        "Uginanie_Nog_Siedzac": {
            "name": "Uginanie Nóg na Maszynie Siedząc (Seated Leg Curl)",
            "primary_target": "dwuglowe_glowa_dluga",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 10.0,
            "muscle_contributions": {
                "dwuglowe_glowa_dluga": 0.7,
                "dwuglowe_glowa_krotka": 0.3
            }
        },
        "Uginanie_Nog_Lezac": {
            "name": "Uginanie Nóg na Maszynie Leżąc (Lying Leg Curl)",
            "primary_target": "dwuglowe_glowa_krotka",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "dwuglowe_glowa_krotka": 0.5,
                "dwuglowe_glowa_dluga": 0.5
            }
        },
        "Uginanie_Nog_Jednonoz": {
            "name": "Uginanie Nóg Stojąc Jednonóż (Standing Leg Curl)",
            "primary_target": "dwuglowe_glowa_krotka",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 8.2,
            "muscle_contributions": {
                "dwuglowe_glowa_krotka": 0.6,
                "dwuglowe_glowa_dluga": 0.4
            }
        },
        "Zuraw_Nordic_Curl": {
            "name": "Żuraw (Nordic Hamstring Curl)",
            "primary_target": "dwuglowe_glowa_dluga",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "MEDIUM",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "dwuglowe_glowa_dluga": 0.6,
                "dwuglowe_glowa_krotka": 0.4
            }
        },
        "Dzien_Dobry_Good_Morning": {
            "name": "Dzień Dobry (Good Morning)",
            "primary_target": "dwuglowe_glowa_dluga",
            "spinal_axial_load": "HIGH",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "dwuglowe_glowa_dluga": 0.6,
                "posladkowy_wielki": 0.5,
                "prostowniki_grzbietu": 0.7
            }
        },

        # =========================================================================
        # 3. MIĘŚNIE POŚLADKOWE (GLUTES)
        # =========================================================================
        "Hip_Thrust_Sztanga": {
            "name": "Wznosy Bioder ze Sztangą (Hip Thrust)",
            "primary_target": "posladkowy_wielki",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "posladkowy_wielki": 0.85,
                "posladkowy_sredni_maly": 0.2,
                "czworoglowe_obszerne": 0.2
            }
        },
        "Glute_Bridge": {
            "name": "Mostki Pośladkowe (Glute Bridge)",
            "primary_target": "posladkowy_wielki",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "posladkowy_wielki": 0.9,
                "posladkowy_sredni_maly": 0.1
            }
        },
        "Hip_Thrust_Jednonoz": {
            "name": "Wznosy Bioder Jednonóż (Single Leg / B-Stance)",
            "primary_target": "posladkowy_wielki",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "posladkowy_wielki": 0.8,
                "posladkowy_sredni_maly": 0.4 # Silna praca stabilizacyjna
            }
        },
        "Odwodzenie_Maszyna_Siedzac": {
            "name": "Odwodzenie Nóg na Maszynie Siedząc (Hip Abductor)",
            "primary_target": "posladkowy_sredni_maly",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "posladkowy_sredni_maly": 0.9,
                "posladkowy_wielki": 0.2
            }
        },
        "Wymachy_Tyl_Kable": {
            "name": "Wymachy Nogi w Tył z Linką Wyciągu (Cable Kickbacks)",
            "primary_target": "posladkowy_wielki",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "posladkowy_wielki": 0.85,
                "posladkowy_sredni_maly": 0.25
            }
        },
        "Odwodzenie_Bok_Kable": {
            "name": "Odwodzenie Nogi w Bok z Linką Wyciągu Dolnego",
            "primary_target": "posladkowy_sredni_maly",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "posladkowy_sredni_maly": 0.95
            }
        },
        "Przeprosty_Lawka_Rzymska_Glute": {
            "name": "Przeprosty na Ławce Rzymskiej (Akcent Pośladki)",
            "primary_target": "posladkowy_wielki",
            "spinal_axial_load": "LOW",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.4,
            "muscle_contributions": {
                "posladkowy_wielki": 0.8,
                "dwuglowe_glowa_dluga": 0.4,
                "prostowniki_grzbietu": 0.3
            }
        },
        "Martwy_Ciag_Sumo": {
            "name": "Martwy Ciąg Sumo",
            "primary_target": "posladkowy_wielki",
            "spinal_axial_load": "HIGH",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "posladkowy_wielki": 0.6,
                "przywodziciel_wielki": 0.6,
                "czworoglowe_obszerne": 0.4,
                "prostowniki_grzbietu": 0.5
            }
        },
        "Monster_Walk_Guma": {
            "name": "Spacer Farmera z Gumą Oporową (Monster Walk)",
            "primary_target": "posladkowy_sredni_maly",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "posladkowy_sredni_maly": 0.9,
                "posladkowy_wielki": 0.2
            }
        },

        # =========================================================================
        # 4. MIĘŚNIE ŁYDEK (CALVES)
        # =========================================================================
        "Wspiecia_Stojac": {
            "name": "Wspięcia na Palce Stojąc (Sztanga / Hantle / Maszyna)",
            "primary_target": "lydka_brzuchaty",
            "spinal_axial_load": "LOW",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "lydka_brzuchaty": 0.8,
                "lydka_plaszczkowaty": 0.3
            }
        },
        "Wspiecia_Siedzac": {
            "name": "Wspięcia na Palce Siedząc na Maszynie",
            "primary_target": "lydka_plaszczkowaty",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "lydka_plaszczkowaty": 0.9,
                "lydka_brzuchaty": 0.1
            }
        },
        "Wspiecia_Suwnica_Leg_Press": {
            "name": "Wspięcia na Palce na Suwnicy (Leg Press)",
            "primary_target": "lydka_brzuchaty",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "lydka_brzuchaty": 0.75,
                "lydka_plaszczkowaty": 0.35
            }
        },
        "Wspiecia_Jednonoz": {
            "name": "Wspięcia na Palce Jednonóż z Obciążeniem",
            "primary_target": "lydka_brzuchaty",
            "spinal_axial_load": "LOW",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "lydka_brzuchaty": 0.8,
                "lydka_plaszczkowaty": 0.3
            }
        },
        "Osle_Wspiecia_Donkey": {
            "name": "Ośle Wspięcia (Donkey Calf Raises)",
            "primary_target": "lydka_brzuchaty",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "lydka_brzuchaty": 0.85,
                "lydka_plaszczkowaty": 0.25
            }
        },

        # =========================================================================
        # 5. MIĘŚNIE PRZYWODZICIELE (ADDUCTORS)
        # =========================================================================
        "Przywodzenie_Maszyna_Siedzac": {
            "name": "Przywodzenie Nóg na Maszynie Siedząc (Hip Adductor)",
            "primary_target": "przywodziciel_wielki",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "przywodziciel_wielki": 0.95
            }
        },
        "Przywodzenie_Kable_Stojac": {
            "name": "Przywodzenie Nogi z Linką Wyciągu Stojąc",
            "primary_target": "przywodziciel_wielki",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "ZERO",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "przywodziciel_wielki": 0.9
            }
        },
        "Przysiad_Sumo_Goblet": {
            "name": "Przysiady z Bardzo Szerokim Rozstawem Nóg (Sumo Squat)",
            "primary_target": "przywodziciel_wielki",
            "spinal_axial_load": "MEDIUM",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "przywodziciel_wielki": 0.7,
                "posladkowy_wielki": 0.5,
                "czworoglowe_obszerne": 0.4
            }
        },
        "Copenhagen_Plank": {
            "name": "Copenhagen Plank (Deska Kopenhaska)",
            "primary_target": "przywodziciel_wielki",
            "spinal_axial_load": "ZERO",
            "knee_shear_force": "LOW",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "przywodziciel_wielki": 0.9
            }
        }
    }

    results = {}

    for key, exercise in legs_database.items():
        # --- ETAP 1: FILTR BEZPIECZEŃSTWA (SAFETY GATE) ---
        is_safe = True
        reason = ""

        if "bol_plecow" in injuries and exercise["spinal_axial_load"] == "HIGH":
            is_safe = False
            reason = "Niebezpieczny nacisk osiowy przy bólach dolnego kręgosłupa."

        if "bol_kolana" in injuries and exercise["knee_shear_force"] == "HIGH":
            is_safe = False
            reason = "Zbyt wysokie siły ścinające w aparacie rzepkowo-udowym."

        if not is_safe:
            results[key] = {
                "name": exercise["name"],
                "status": "DISQUALIFIED",
                "score": 0.0,
                "reason": reason,
                "muscle_contributions": {}
            }
            continue

        # --- ETAP 2: OPTYMALIZACJA HIPERTROFII POD DŹWIGNIE ---
        score = exercise["base_hypertrophy_score"]

        # Długa kość udowa (femur_torso_ratio > 0.85)
        if femur_torso_ratio > 0.85:
            if "High_Bar" in key or "Low_Bar" in key or "Good_Morning" in key:
                score -= 1.5 # Długie udo obniża efektywność na czwórki i przeciąża plecy
            elif "Pendulum" in key or "Hack" in key or "Belt" in key:
                score = min(10.0, score + 0.3)

        results[key] = {
            "name": exercise["name"],
            "status": "APPROVED",
            "score": round(score, 1),
            "primary_target": exercise["primary_target"],
            "muscle_contributions": exercise["muscle_contributions"]
        }

    return results


# --- TEST WARSZTATOWY ---
if __name__ == "__main__":
    import json

    test_biometrics = {"femur_torso_ratio": 0.88, "tibia_femur_ratio": 0.80}
    test_injuries = ["bol_plecow"]

    analysis = analyze_legs_exercises(test_biometrics, test_injuries)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))