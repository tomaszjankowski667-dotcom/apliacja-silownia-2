"""
ABS MODULE (abs.py)
-------------------
Pełny atlas ćwiczeń mięśni brzucha i kompleksu core:
1. PROSTY BRZUCHA - CZĘŚĆ DOLNA (brzuch_prosty_dol)
2. PROSTY BRZUCHA - CZĘŚĆ GÓRNA (brzuch_prosty_gora)
3. SKOŚNE BRZUCHA (brzuch_skosne)
4. POPRZECZNY I CORE / STABILIZACJA GŁĘBOKA (brzuch_poprzeczny_core)
"""

def analyze_abs_exercises(biometrics: dict, injuries: list) -> dict:
    torso_length = biometrics.get("torso_length", 1.0)

    abs_database = {
        # =========================================================================
        # 1. MIĘŚNIE PROSTE BRZUCHA - CZĘŚĆ DOLNA (LOWER ABS - PODWIJANIE MIEDNICY)
        # =========================================================================
        "Unoszenie_Nog_Wiszenie_Drazek": {
            "name": "Unoszenie Nóg / Kolan w Wiszeniu na Drążku",
            "primary_target": "brzuch_prosty_dol",
            "lumbar_strain_risk": "MEDIUM", # Ryzyko jeśli zginacze bioder ciągną lędźwie przy braku podwijania
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "brzuch_prosty_dol": 0.8,
                "brzuch_prosty_gora": 0.3,
                "brzuch_skosne": 0.2,
                "zginacze_bioder": 0.4
            }
        },
        "Unoszenie_Nog_Porecze_Kapitanska": {
            "name": "Unoszenie Kolan / Nóg w Podparciu na Poręczach (Kapitańska Stolica)",
            "primary_target": "brzuch_prosty_dol",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "brzuch_prosty_dol": 0.75,
                "brzuch_prosty_gora": 0.3,
                "zginacze_bioder": 0.3
            }
        },
        "Odwrotne_Brzuszki_Lawka_Skosna": {
            "name": "Odwrotne Brzuszki na Ławce Skośnej Ujemnej (Reverse Crunches)",
            "primary_target": "brzuch_prosty_dol",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "brzuch_prosty_dol": 0.8,
                "brzuch_prosty_gora": 0.2
            }
        },
        "Odwrotne_Brzuszki_Podloga_Dociazenie": {
            "name": "Odwrotne Brzuszki na Podłodze z Dociążeniem na Kostkach",
            "primary_target": "brzuch_prosty_dol",
            "lumbar_strain_risk": "MEDIUM",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "brzuch_prosty_dol": 0.85,
                "brzuch_prosty_gora": 0.15
            }
        },
        "Scissor_Flutter_Kicks_Ciezar": {
            "name": "Scissor Kicks / Flutter Kicks z Obciążeniem",
            "primary_target": "brzuch_prosty_dol",
            "lumbar_strain_risk": "HIGH", # Duża siła zginaczy bioder ciągnąca kręgosłup do lordozy
            "base_hypertrophy_score": 7.0,
            "muscle_contributions": {
                "brzuch_prosty_dol": 0.6,
                "zginacze_bioder": 0.6
            }
        },

        # =========================================================================
        # 2. MIĘŚNIE PROSTE BRZUCHA - CZĘŚĆ GÓRNA (UPPER ABS - ŻEBRA DO MIEDNICY)
        # =========================================================================
        "Brzuszki_Wyciag_Kleczac_Allahy": {
            "name": "Brzuszki na Wyciągu Klęcząc (Allahy / Cable Crunches)",
            "primary_target": "brzuch_prosty_gora",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 10.0, # Król progresji ciężaru i izolacji
            "muscle_contributions": {
                "brzuch_prosty_gora": 0.85,
                "brzuch_prosty_dol": 0.3,
                "brzuch_poprzeczny_core": 0.2
            }
        },
        "Brzuszki_Lawka_Skosna_Ujemna": {
            "name": "Brzuszki na Ławce Skośnej Ujemnej z Talerzem / Hantlem (Decline Sit-Ups)",
            "primary_target": "brzuch_prosty_gora",
            "lumbar_strain_risk": "MEDIUM",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "brzuch_prosty_gora": 0.65,
                "zginacze_bioder": 0.5,
                "brzuch_prosty_dol": 0.2
            }
        },
        "Brzuszki_Maszyna_Ab_Crunch": {
            "name": "Brzuszki na Maszynie (Ab Crunch Machine)",
            "primary_target": "brzuch_prosty_gora",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "brzuch_prosty_gora": 0.8,
                "brzuch_prosty_dol": 0.2
            }
        },
        "Sklony_Weighted_Crunches": {
            "name": "Skłony Tułowia z Hantlem w Leżeniu Tyłem (Weighted Crunches)",
            "primary_target": "brzuch_prosty_gora",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "brzuch_prosty_gora": 0.85,
                "brzuch_prosty_dol": 0.15
            }
        },
        "Spiecia_Pilka_Gimnastyczna": {
            "name": "Spięcia na Piłce Gimnastycznej z Ciężarem",
            "primary_target": "brzuch_prosty_gora",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 9.6, # Kapitalne rozciągnięcie na krzywiźnie piłki
            "muscle_contributions": {
                "brzuch_prosty_gora": 0.8,
                "brzuch_prosty_dol": 0.3
            }
        },

        # =========================================================================
        # 3. MIĘŚNIE SKOŚNE BRZUCHA (OBLIQUES)
        # =========================================================================
        "Woodchoppers_Wyciag": {
            "name": "Rotacje Tułowia na Wyciągu (Woodchoppers)",
            "primary_target": "brzuch_skosne",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "brzuch_skosne": 0.85,
                "brzuch_poprzeczny_core": 0.3
            }
        },
        "Rosyjski_Skret_Russian_Twist": {
            "name": "Rosyjski Skręt Tułowia z Talerzem / Hantlem (Weighted Russian Twist)",
            "primary_target": "brzuch_skosne",
            "lumbar_strain_risk": "MEDIUM",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "brzuch_skosne": 0.75,
                "brzuch_prosty_gora": 0.25
            }
        },
        "Sklony_Boczne_Lawka_Rzymska_Hantel": {
            "name": "Skłony Boczne na Ławce Rzymskiej / Podłodze z Hantlem (Side Bends)",
            "primary_target": "brzuch_skosne",
            "lumbar_strain_risk": "MEDIUM",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "brzuch_skosne": 0.9,
                "czworoboczny_ledzwi": 0.4
            }
        },
        "Spiecia_Z_Rotacja_Lezenie": {
            "name": "Spięcia Brzucha z Rotacją w Leżeniu (Oblique Crunches)",
            "primary_target": "brzuch_skosne",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 7.8,
            "muscle_contributions": {
                "brzuch_skosne": 0.7,
                "brzuch_prosty_gora": 0.3
            }
        },
        "Windshield_Wipers_Wiszenie": {
            "name": "Wznosy Nóg w Wiszeniu z Przeniesieniem na Boki (Windshield Wipers)",
            "primary_target": "brzuch_skosne",
            "lumbar_strain_risk": "HIGH",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "brzuch_skosne": 0.8,
                "brzuch_prosty_dol": 0.4,
                "sila_chwytu_przedramie": 0.3
            }
        },

        # =========================================================================
        # 4. MIĘŚEŃ POPRZECZNY I CORE (TRANSVERSE ABDOMINIS & CORE)
        # =========================================================================
        "Kolko_Ab_Wheel_Roller": {
            "name": "Kółko do Brzucha (Ab Wheel / Ab Roller)",
            "primary_target": "brzuch_poprzeczny_core",
            "lumbar_strain_risk": "HIGH", # Wymaga mocnej kontroli miednicy, inaczej wpada w wyprost L5-S1
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "brzuch_poprzeczny_core": 0.8,
                "brzuch_prosty_gora": 0.6,
                "brzuch_prosty_dol": 0.5,
                "najszerszy_pionowy": 0.3
            }
        },
        "Deska_Weighted_Plank": {
            "name": "Deska z Dociążeniem na Plecach (Weighted Plank)",
            "primary_target": "brzuch_poprzeczny_core",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "brzuch_poprzeczny_core": 0.85,
                "brzuch_prosty_gora": 0.3,
                "brzuch_prosty_dol": 0.3
            }
        },
        "Suitcase_Carry_Farmer_Jednoracz": {
            "name": "Spacer Farmera Jednorącz (Suitcase Carry)",
            "primary_target": "brzuch_poprzeczny_core",
            "lumbar_strain_risk": "LOW",
            "base_hypertrophy_score": 9.2, # Wzmocnienie anty-zgięcia bocznego
            "muscle_contributions": {
                "brzuch_poprzeczny_core": 0.75,
                "brzuch_skosne": 0.7,
                "czworoboczny_ledzwi": 0.5,
                "sila_chwytu_przedramie": 0.5
            }
        },
        "Proznia_Brzuszna_Stomach_Vacuum": {
            "name": "Próżnia Brzuszna (Stomach Vacuum)",
            "primary_target": "brzuch_poprzeczny_core",
            "lumbar_strain_risk": "ZERO",
            "base_hypertrophy_score": 7.0, # Ćwiczenie kontroli i taliowania, brak bezpośredniej hipertrofii pod oporem
            "muscle_contributions": {
                "brzuch_poprzeczny_core": 1.0
            }
        }
    }

    results = {}

    for key, exercise in abs_database.items():
        # --- ETAP 1: FILTR BEZPIECZEŃSTWA (SAFETY GATE) ---
        is_safe = True
        disqualification_reason = ""

        # Kontuzja / Bolesność odcinka lędźwiowego
        if "bol_plecow" in injuries or "dyskopatia_L5_S1" in injuries:
            if exercise["lumbar_strain_risk"] == "HIGH":
                is_safe = False
                disqualification_reason = "Kwarantanna: Brak dostatecznej stabilizacji lędźwiowej / silny hiperektensyjny ciąg zginaczy bioder."

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

    test_biometrics = {"torso_length": 1.0}
    test_injuries = ["bol_plecow"]

    analysis = analyze_abs_exercises(test_biometrics, test_injuries)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))