"""
BACK MODULE (back.py)
---------------------
Pełny atlas ćwiczeń pleców z podziałem na 4 obszary anatomiczne:
1. NAJSZERSZY GRZBIETU (najszerszy_pionowy, najszerszy_poziomy)
2. ŚRODEK I GÓRA PLECÓW (obly_wiekszy, czworoboczny_srodek_dol, czworoboczny_gora_kaptur)
3. RÓWNOLEGŁOBOCZNE (rownolegloboczny)
4. PROSTOWNIKI KRĘGOSŁUPA (prostowniki_grzbietu)
"""


def analyze_back_exercises(biometrics: dict, injuries: list) -> dict:
    reach_ratio = biometrics.get("reach_ratio", 1.0)
    torso_length = biometrics.get("torso_length", 1.0)

    back_database = {
        # =========================================================================
        # 1. MIĘŚNIE NAJSZERSZE GRZBIETU - RUCHY PIONOWE (WIDTH / LATS)
        # =========================================================================
        "Sciaganie_Drazka_Wyciag_Gora": {
            "name": "Ściąganie Drążka Wyciągu Górnego do Klatki",
            "primary_target": "najszerszy_pionowy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "najszerszy_pionowy": 0.8,
                "obly_wiekszy": 0.5,
                "biceps": 0.3,
                "czworoboczny_srodek_dol": 0.2
            }
        },
        "Podciaganie_Na_Drazku": {
            "name": "Podciąganie na Drążku (Nachwyt / Podchwyt / Neutralny)",
            "primary_target": "najszerszy_pionowy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "najszerszy_pionowy": 0.75,
                "obly_wiekszy": 0.5,
                "biceps": 0.4,
                "czworoboczny_srodek_dol": 0.3
            }
        },
        "Sciaganie_Wyciag_Gora_Jednoracz": {
            "name": "Ściąganie Drążka / Uchwytów Wyciągu Górnego Jednorącz",
            "primary_target": "najszerszy_pionowy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.8,  # Świetne dopasowanie do toru włókien
            "muscle_contributions": {
                "najszerszy_pionowy": 0.85,
                "obly_wiekszy": 0.4,
                "biceps": 0.25
            }
        },
        "Lat_Pulldown_Maszyna_Hammer": {
            "name": "Ściąganie Drążka na Maszynie Lat Pulldown (np. Hammer)",
            "primary_target": "najszerszy_pionowy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.6,
            "muscle_contributions": {
                "najszerszy_pionowy": 0.8,
                "obly_wiekszy": 0.45,
                "biceps": 0.2
            }
        },

        # =========================================================================
        # 1. MIĘŚNIE NAJSZERSZE GRZBIETU - RUCHY POZIOME I PRZENOSZENIA (THICKNESS & LOWER LATS)
        # =========================================================================
        "Wioslowanie_Hantlem_Lawka_Jednoracz": {
            "name": "Wiosłowanie Hantlem w Podparciu o Ławkę (Jednorącz)",
            "primary_target": "najszerszy_poziomy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "najszerszy_poziomy": 0.8,
                "czworoboczny_srodek_dol": 0.3,
                "rownolegloboczny": 0.3,
                "biceps": 0.3
            }
        },
        "Wioslowanie_Wyciag_Dolny_Wasko": {
            "name": "Wiosłowanie na Wyciągu Dolnym do Brzucha (Chwyt Neutralny / Wąski)",
            "primary_target": "najszerszy_poziomy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "najszerszy_poziomy": 0.75,
                "czworoboczny_srodek_dol": 0.4,
                "rownolegloboczny": 0.3,
                "biceps": 0.25
            }
        },
        "Przenoszenie_Straight_Arm_Pulldown": {
            "name": "Przenoszenie Drążka / Liny na Wyciągu Górnym na Prostych Rękach",
            "primary_target": "najszerszy_pionowy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.2,  # Izolacja bez udziału bicepsa
            "muscle_contributions": {
                "najszerszy_pionowy": 0.85,
                "obly_wiekszy": 0.4,
                "triceps_glowa_dluga": 0.3
            }
        },
        "Wioslowanie_Maszyna_Jednoracz_Blisko": {
            "name": "Wiosłowanie na Maszynie Jednorącz (Łokieć Blisko Tułowia)",
            "primary_target": "najszerszy_poziomy",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "najszerszy_poziomy": 0.85,
                "czworoboczny_srodek_dol": 0.2,
                "biceps": 0.2
            }
        },

        # =========================================================================
        # 2. ŚRODEK I GÓRA PLECÓW (UPPER BACK & TRAPS)
        # =========================================================================
        "Wioslowanie_Sztanga_Opad": {
            "name": "Wiosłowanie Sztangą w Opadzie Tułowia",
            "primary_target": "czworoboczny_srodek_dol",
            "spinal_axial_load": "HIGH",
            "lower_back_shear": "HIGH",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "czworoboczny_srodek_dol": 0.6,
                "rownolegloboczny": 0.5,
                "najszerszy_poziomy": 0.4,
                "prostowniki_grzbietu": 0.6,
                "biceps": 0.3
            }
        },
        "Wioslowanie_Polsztanga_T_Bar": {
            "name": "Wiosłowanie Półsztangą / Chwytem T (T-Bar Row)",
            "primary_target": "czworoboczny_srodek_dol",
            "spinal_axial_load": "HIGH",
            "lower_back_shear": "HIGH",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "czworoboczny_srodek_dol": 0.65,
                "rownolegloboczny": 0.5,
                "najszerszy_poziomy": 0.4,
                "prostowniki_grzbietu": 0.5
            }
        },
        "Seal_Row_Wyciag_Szeroko": {
            "name": "Wiosłowanie na Wyciągu Poziomym Szeroko z Odchyleniem Łokci (Seal / Cable Row)",
            "primary_target": "czworoboczny_srodek_dol",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.6,
            "muscle_contributions": {
                "czworoboczny_srodek_dol": 0.75,
                "rownolegloboczny": 0.6,
                "obly_wiekszy": 0.4,
                "bark_tyl": 0.4
            }
        },
        "Wioslowanie_Meadows_Chest_Supported": {
            "name": "Wiosłowanie Hantlami na Ławce Skośnej Przodem (Chest-Supported Row)",
            "primary_target": "czworoboczny_srodek_dol",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.8,  # Brak zmęczenia dolnych pleców
            "muscle_contributions": {
                "czworoboczny_srodek_dol": 0.75,
                "rownolegloboczny": 0.6,
                "obly_wiekszy": 0.4,
                "bark_tyl": 0.3
            }
        },
        "Face_Pull_Lina": {
            "name": "Przyciąganie Liny do Twarzy (Face Pull)",
            "primary_target": "czworoboczny_srodek_dol",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "czworoboczny_srodek_dol": 0.6,
                "bark_tyl": 0.5,
                "rotatory_zewnetrzne": 0.4
            }
        },

        # --- GÓRNY AKTON CZWOROBOCZNEGO (KAPTURY / UPPER TRAPS) ---
        "Szrugsy_Hantlis": {
            "name": "Szrugsy / Wznosy Barków z Hantlami (Stojąc / Siedząc)",
            "primary_target": "czworoboczny_gora_kaptur",
            "spinal_axial_load": "MEDIUM",
            "lower_back_shear": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "czworoboczny_gora_kaptur": 0.95
            }
        },
        "Szrugsy_Sztanga": {
            "name": "Szrugsy ze Sztangą (z Przodu lub z Tyłu)",
            "primary_target": "czworoboczny_gora_kaptur",
            "spinal_axial_load": "HIGH",
            "lower_back_shear": "MEDIUM",
            "base_hypertrophy_score": 8.2,
            "muscle_contributions": {
                "czworoboczny_gora_kaptur": 0.9,
                "prostowniki_grzbietu": 0.3
            }
        },
        "Szrugsy_Smith_Maszyna": {
            "name": "Szrugsy na Maszynie Smitha / Maszynie do Wspięć",
            "primary_target": "czworoboczny_gora_kaptur",
            "spinal_axial_load": "MEDIUM",
            "lower_back_shear": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "czworoboczny_gora_kaptur": 0.95
            }
        },
        "Szrugsy_Kable_Wyciag": {
            "name": "Szrugsy z Linkami Wyciągu Dolnego",
            "primary_target": "czworoboczny_gora_kaptur",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.6,  # Wektor oporu idealnie po skosie pod kątem włókien
            "muscle_contributions": {
                "czworoboczny_gora_kaptur": 0.95
            }
        },

        # =========================================================================
        # 3. MIĘŚNIE RÓWNOLEGŁOBOCZNE (RHOMBOIDS)
        # =========================================================================
        "Wioslowanie_Maszyna_Podparcie_Szeroko": {
            "name": "Wiosłowanie na Maszynie z Podparciem Klatki i Szerokim Chwytem",
            "primary_target": "rownolegloboczny",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.6,
            "muscle_contributions": {
                "rownolegloboczny": 0.8,
                "czworoboczny_srodek_dol": 0.6,
                "bark_tyl": 0.3
            }
        },
        "Wioslowanie_Hantle_Opad_Pauza": {
            "name": "Wiosłowanie Sztangielkami w Opadzie Tułowia z Pauzą w Spięciu",
            "primary_target": "rownolegloboczny",
            "spinal_axial_load": "MEDIUM",
            "lower_back_shear": "MEDIUM",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "rownolegloboczny": 0.75,
                "czworoboczny_srodek_dol": 0.5,
                "prostowniki_grzbietu": 0.4
            }
        },
        "Odwrotne_Rozpietki_Akcent_Lopatki": {
            "name": "Odwrotne Rozpiętki na Wyciągu / Maszynie (Akcent Ściąganie Łopatek)",
            "primary_target": "rownolegloboczny",
            "spinal_axial_load": "ZERO",
            "lower_back_shear": "ZERO",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "rownolegloboczny": 0.7,
                "czworoboczny_srodek_dol": 0.6,
                "bark_tyl": 0.4
            }
        },

        # =========================================================================
        # 4. PROSTOWNIKI KRĘGOSŁUPA (ERECTOR SPINAE)
        # =========================================================================
        "Martwy_Ciag_Klasyczny": {
            "name": "Martwy Ciąg Klasyczny (Conventional Deadlift)",
            "primary_target": "prostowniki_grzbietu",
            "spinal_axial_load": "HIGH",
            "lower_back_shear": "HIGH",
            "base_hypertrophy_score": 7.5,  # Ogromne obciążenie CUN, trudna regeneracja pod hipertrofię
            "muscle_contributions": {
                "prostowniki_grzbietu": 0.8,
                "posladkowy_wielki": 0.6,
                "dwuglowe_glowa_dluga": 0.5,
                "przywodziciel_wielki": 0.4,
                "czworoboczny_gora_kaptur": 0.5
            }
        },
        "Martwy_Ciag_Rack_Pull": {
            "name": "Martwy Ciąg z Podwyższenia (Rack Pull)",
            "primary_target": "prostowniki_grzbietu",
            "spinal_axial_load": "HIGH",
            "lower_back_shear": "HIGH",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "prostowniki_grzbietu": 0.75,
                "czworoboczny_gora_kaptur": 0.6,
                "posladkowy_wielki": 0.4
            }
        },
        "Wyprosty_Lawka_Rzymska_45": {
            "name": "Wyprosty Tułowia na Ławce Rzymskiej (45° Back Extension)",
            "primary_target": "prostowniki_grzbietu",
            "spinal_axial_load": "LOW",
            "lower_back_shear": "MEDIUM",
            "base_hypertrophy_score": 9.5,  # Bezpieczny i bardzo efektywny bodziec
            "muscle_contributions": {
                "prostowniki_grzbietu": 0.85,
                "posladkowy_wielki": 0.5,
                "dwuglowe_glowa_dluga": 0.4
            }
        },
        "Dzien_Dobry_Good_Morning": {
            "name": "Dzień Dobry ze Sztangą (Good Morning)",
            "primary_target": "prostowniki_grzbietu",
            "spinal_axial_load": "HIGH",
            "lower_back_shear": "HIGH",
            "base_hypertrophy_score": 7.0,
            "muscle_contributions": {
                "prostowniki_grzbietu": 0.8,
                "dwuglowe_glowa_dluga": 0.6,
                "posladkowy_wielki": 0.5
            }
        },
        "Przysiad_Sztanga_Kark_Stabilizacja": {
            "name": "Przysiad ze Sztangą na Karku (Stabilizacja Prostowników)",
            "primary_target": "prostowniki_grzbietu",
            "spinal_axial_load": "HIGH",
            "lower_back_shear": "MEDIUM",
            "base_hypertrophy_score": 6.5,  # Funkcja czysto stabilizacyjna dla pleców
            "muscle_contributions": {
                "prostowniki_grzbietu": 0.5,
                "czworoglowe_obszerne": 0.8,
                "posladkowy_wielki": 0.6
            }
        }
    }

    results = {}

    for key, exercise in back_database.items():
        # --- ETAP 1: FILTR BEZPIECZEŃSTWA (SAFETY GATE) ---
        is_safe = True
        disqualification_reason = ""

        # Ból / Kontuzja kręgosłupa (L5-S1, dyskopatia)
        if "bol_plecow" in injuries or "dyskopatia_L5_S1" in injuries:
            if exercise["spinal_axial_load"] == "HIGH" or exercise["lower_back_shear"] == "HIGH":
                is_safe = False
                disqualification_reason = "Kwarantanna: Niebezpieczny nacisk osiowy lub siły ścinające w dolnym kręgosłupie."

        if not is_safe:
            results[key] = {
                "name": exercise["name"],
                "status": "DISQUALIFIED",
                "score": 0.0,
                "reason": disqualification_reason,
                "muscle_contributions": {}
            }
            continue

        # --- ETAP 2: OPTYMALIZACJA HIPERTROFII UNDER LEVERAGE ---
        score = exercise["base_hypertrophy_score"]

        # Długie ramiona (reach_ratio > 1.2) - idealne do ciągów i wyciągów
        if reach_ratio > 1.2:
            if "Wyciag" in key or "Maszyna" in key or "Pulldown" in key:
                score = min(10.0, score + 0.3)

        results[key] = {
            "name": exercise["name"],
            "status": "APPROVED",
            "score": round(score, 1),
            "primary_target": exercise["primary_target"],
            "muscle_contributions": exercise["muscle_contributions"]
        }

    return results


# --- TEST WARSZTATOWY MODUŁU ---
if __name__ == "__main__":
    import json

    test_biometrics = {"reach_ratio": 1.25, "torso_length": 1.1}
    test_injuries = ["bol_plecow"]

    analysis = analyze_back_exercises(test_biometrics, test_injuries)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))