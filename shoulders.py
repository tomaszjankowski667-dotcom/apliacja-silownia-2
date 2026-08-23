"""
SHOULDERS MODULE (shoulders.py)
-------------------------------
Pełny atlas ćwiczeń mięśni naramiennych z podziałem na 3 anatomiczne aktony:
1. PRZEDNI AKTON / CZĘŚĆ OBOJCZYKOWA (bark_przod)
2. BOCZNY AKTON / CZĘŚĆ BARKOWA (bark_bok)
3. TYLNY AKTON / CZĘŚĆ GRZEBIENIOWA (bark_tyl)
"""


def analyze_shoulders_exercises(biometrics: dict, injuries: list) -> dict:
    reach_ratio = biometrics.get("reach_ratio", 1.0)

    shoulders_database = {
        # =========================================================================
        # 1. PRZEDNI AKTON (ANTERIOR DELTOID)
        # =========================================================================
        "OHP_Zolnierskie_Sztanga_Stojac": {
            "name": "Wyciskanie Żołnierskie ze Sztangą Stojąc (OHP)",
            "primary_target": "bark_przod",
            "impingement_risk": "MEDIUM",
            "spinal_axial_load": "HIGH",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "bark_przod": 0.7,
                "bark_bok": 0.2,
                "triceps_glowa_boczna_przysrodkowa": 0.4,
                "klatka_gora_obojczyk": 0.3,
                "prostowniki_grzbietu": 0.3
            }
        },
        "Wyciskanie_Hantli_Barki": {
            "name": "Wyciskanie Hantli na Barki (Siedząc / Stojąc)",
            "primary_target": "bark_przod",
            "impingement_risk": "LOW",
            "spinal_axial_load": "MEDIUM",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "bark_przod": 0.75,
                "bark_bok": 0.25,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Wyciskanie_Smith_Nad_Glowe": {
            "name": "Wyciskanie na Maszynie Smitha Nad Głowę (Siedząc)",
            "primary_target": "bark_przod",
            "impingement_risk": "LOW",
            "spinal_axial_load": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "bark_przod": 0.8,
                "bark_bok": 0.2,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Wyciskanie_Hammer_Barki": {
            "name": "Wyciskanie na Maszynie typu Hammer na Barki",
            "primary_target": "bark_przod",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "bark_przod": 0.85,
                "bark_bok": 0.15,
                "triceps_glowa_boczna_przysrodkowa": 0.3
            }
        },
        "Wyciskanie_Arnolda_Arnold_Press": {
            "name": "Wyciskanie Arnolda (Arnold Press)",
            "primary_target": "bark_przod",
            "impingement_risk": "MEDIUM",
            "spinal_axial_load": "MEDIUM",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "bark_przod": 0.7,
                "bark_bok": 0.3,
                "triceps_glowa_boczna_przysrodkowa": 0.3
            }
        },
        "Wznosy_Przod_Hantlis": {
            "name": "Wznosy Ramion w Przód z Hantlami",
            "primary_target": "bark_przod",
            "impingement_risk": "LOW",
            "spinal_axial_load": "LOW",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "bark_przod": 0.9,
                "klatka_gora_obojczyk": 0.2
            }
        },
        "Wznosy_Przod_Sztanga_Talerz": {
            "name": "Wznosy Ramion w Przód ze Sztangą lub Talerzem",
            "primary_target": "bark_przod",
            "impingement_risk": "LOW",
            "spinal_axial_load": "LOW",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "bark_przod": 0.85,
                "klatka_gora_obojczyk": 0.2
            }
        },
        "Wznosy_Przod_Kable": {
            "name": "Wznosy Ramion w Przód z Linką Wyciągu Dolnego",
            "primary_target": "bark_przod",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.0,  # Wyciąg zapewnia stałe napięcie w rozciągnięciu
            "muscle_contributions": {
                "bark_przod": 0.95,
                "klatka_gora_obojczyk": 0.15
            }
        },

        # =========================================================================
        # 2. BOCZNY AKTON (LATERAL DELTOID)
        # =========================================================================
        "Wznosy_Bok_Hantlis": {
            "name": "Wznosy Ramion Bokiem z Hantlami (Stojąc / Siedząc)",
            "primary_target": "bark_bok",
            "impingement_risk": "MEDIUM",
            "spinal_axial_load": "LOW",
            "base_hypertrophy_score": 8.0,  # Brak oporu w dolnej fazie ruchu
            "muscle_contributions": {
                "bark_bok": 0.85,
                "czworoboczny_gora": 0.2,
                "bark_przod": 0.1
            }
        },
        "Wznosy_Bok_Kable": {
            "name": "Wznosy Ramion Bokiem z Linką Wyciągu Dolnego",
            "primary_target": "bark_bok",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.8,  # Idealny profil oporu od samego dołu
            "muscle_contributions": {
                "bark_bok": 0.95,
                "czworoboczny_gora": 0.1
            }
        },
        "Wznosy_Bok_Lawka_Skosna": {
            "name": "Wznosy Ramion Bokiem w Leżeniu Bokiem na Ławce Skośnej",
            "primary_target": "bark_bok",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.2,  # Przesunięcie szczytu oporu na pełne rozciągnięcie
            "muscle_contributions": {
                "bark_bok": 0.9,
                "czworoboczny_gora": 0.1
            }
        },
        "Wznosy_Bok_Maszyna": {
            "name": "Wznosy Ramion Bokiem na Maszynie (Lateral Raise Machine)",
            "primary_target": "bark_bok",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.6,
            "muscle_contributions": {
                "bark_bok": 0.95,
                "czworoboczny_gora": 0.1
            }
        },
        "Podciaganie_Upright_Row": {
            "name": "Podciąganie Sztangi / Hantli / Wyciągu wzdłuż Tułowia (Upright Row)",
            "primary_target": "bark_bok",
            "impingement_risk": "HIGH",  # Połączenie rotacji wewnętrznej i wysokiego odwiedzenia
            "spinal_axial_load": "MEDIUM",
            "base_hypertrophy_score": 7.0,
            "muscle_contributions": {
                "bark_bok": 0.7,
                "czworoboczny_gora": 0.5,
                "biceps": 0.3
            }
        },
        "Y_Raise_Kable_Lawka": {
            "name": "Wznosy Y-Raise z Linkami Wyciągu lub Hantlami na Ławce",
            "primary_target": "bark_bok",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.4,
            "muscle_contributions": {
                "bark_bok": 0.75,
                "bark_tyl": 0.25,
                "czworoboczny_srodek_dol": 0.3
            }
        },

        # =========================================================================
        # 3. TYLNY AKTON (POSTERIOR DELTOID)
        # =========================================================================
        "Odwrotne_Rozpietki_Peck_Deck": {
            "name": "Odwrotne Rozpiętki na Maszynie Butterfly (Reverse Peck-Deck)",
            "primary_target": "bark_tyl",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "bark_tyl": 0.85,
                "czworoboczny_srodek_dol": 0.3,
                "rownolegloboczny": 0.3
            }
        },
        "Odwrotne_Rozpietki_Kable_Cross": {
            "name": "Odwrotne Rozpiętki z Linkami Wyciągu Górnego (Reverse Cross-Cable)",
            "primary_target": "bark_tyl",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 10.0,  # Idealna linia pracy zgodna z włóknami
            "muscle_contributions": {
                "bark_tyl": 0.95,
                "czworoboczny_srodek_dol": 0.2
            }
        },
        "Wznosy_Tyl_Hantles_Opad": {
            "name": "Wznosy Hantli w Opadzie Tułowia (Stojąc / Siedząc)",
            "primary_target": "bark_tyl",
            "impingement_risk": "LOW",
            "spinal_axial_load": "MEDIUM",
            "base_hypertrophy_score": 7.8,
            "muscle_contributions": {
                "bark_tyl": 0.8,
                "czworoboczny_srodek_dol": 0.3,
                "prostowniki_grzbietu": 0.3
            }
        },
        "Wznosy_Tyl_Lawka_Przodem": {
            "name": "Wznosy Hantli w Leżeniu Przodem na Ławce Skośnej",
            "primary_target": "bark_tyl",
            "impingement_risk": "LOW",
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "bark_tyl": 0.85,
                "czworoboczny_srodek_dol": 0.25
            }
        },
        "Face_Pull_Lina": {
            "name": "Przyciąganie Lina Wyciągu do Twarzy (Face Pull)",
            "primary_target": "bark_tyl",
            "impingement_risk": "LOW",  # Bardzo bezpieczne, wzmacnia rotatory zewnętrzne
            "spinal_axial_load": "ZERO",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "bark_tyl": 0.7,
                "rotatory_zewnetrzne": 0.5,
                "czworoboczny_srodek_dol": 0.4
            }
        },
        "Przyciaganie_Szeroko_Opad": {
            "name": "Przyciąganie Sztangi / Hantli do Klatki w Szerokim Chwycie (Opad)",
            "primary_target": "bark_tyl",
            "impingement_risk": "LOW",
            "spinal_axial_load": "MEDIUM",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "bark_tyl": 0.75,
                "czworoboczny_srodek_dol": 0.4,
                "najszerszy_grzbietu": 0.2
            }
        }
    }

    results = {}

    for key, exercise in shoulders_database.items():
        # --- ETAP 1: FILTR BEZPIECZEŃSTWA (SAFETY GATE) ---
        is_safe = True
        disqualification_reason = ""

        if "kontuzja_barku" in injuries:
            if exercise["impingement_risk"] == "HIGH":
                is_safe = False
                disqualification_reason = "Niebezpieczna rotacja wewnętrzna przy odwiedzeniu (Ryzyko konfliktu podbarkowego)."
            elif exercise["impingement_risk"] == "MEDIUM" and "ostry_stan_zapalny" in injuries:
                is_safe = False
                disqualification_reason = "Wysokie obciążenie stożka rotatorów w ostrym stanie zapalnym."

        if "bol_plecow" in injuries and exercise["spinal_axial_load"] == "HIGH":
            is_safe = False
            disqualification_reason = "Przekroczony bezpieczny próg nacisku osiowego na kręgosłup."

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

        # Długie ramiona (reach_ratio > 1.2) premiują wyciągi i maszyny ze stałym ramieniem siły
        if reach_ratio > 1.2:
            if "Kable" in key or "Maszyna" in key or "Peck_Deck" in key:
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

    test_biometrics = {"reach_ratio": 1.25}
    test_injuries = ["kontuzja_barku"]

    analysis = analyze_shoulders_exercises(test_biometrics, test_injuries)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))