"""
CHEST MODULE (chest.py)
-----------------------
Pełny atlas ćwiczeń klatki piersiowej z podziałem na 3 anatomiczne aktony:
1. CZĘŚĆ GÓRNA / OBOJCZYKOWA (klatka_gora_obojczyk)
2. CZĘŚĆ ŚRODKOWA / MOSTKOWO-ŻEBROWA (klatka_srodek_mostek)
3. CZĘŚĆ DOLNA / BRZUSZNA (klatka_dol_brzuszna)
"""

def analyze_chest_exercises(biometrics: dict, injuries: list) -> dict:
    reach_ratio = biometrics.get("reach_ratio", 1.0)

    chest_database = {
        # =========================================================================
        # 1. CZĘŚĆ GÓRNA / OBOJCZYKOWA (UPPER / CLAVICULAR)
        # =========================================================================
        "Wyciskanie_Hantli_Skos_Dodatni": {
            "name": "Wyciskanie Hantli na Ławce Skośnej Dodatniej",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 50,
            "deep_extension_risk": "MEDIUM",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.7,
                "klatka_srodek_mostek": 0.3,
                "bark_przod": 0.5,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Wyciskanie_Sztangi_Skos_Dodatni": {
            "name": "Wyciskanie Sztangi na Ławce Skośnej Dodatniej",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 70,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.65,
                "klatka_srodek_mostek": 0.35,
                "bark_przod": 0.6,
                "triceps_glowa_boczna_przysrodkowa": 0.5
            }
        },
        "Wyciskanie_Smith_Skos_Dodatni": {
            "name": "Wyciskanie na Maszynie Smitha na Skosie Dodatnim",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 50,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.75,
                "klatka_srodek_mostek": 0.25,
                "bark_przod": 0.4,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Wyciskanie_Hammer_Skos_Dodatni": {
            "name": "Wyciskanie na Maszynie Hammer na Skos Dodatni",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 45,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.8,
                "klatka_srodek_mostek": 0.2,
                "bark_przod": 0.3,
                "triceps_glowa_boczna_przysrodkowa": 0.3
            }
        },
        "Rozpietki_Kable_Dol_Gora": {
            "name": "Rozpiętki z Linkami Wyciągu z Dołu do Góry",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 45,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.85,
                "klatka_srodek_mostek": 0.15,
                "bark_przod": 0.4
            }
        },
        "Rozpietki_Hantle_Skos_Dodatni": {
            "name": "Rozpiętki z Hantlami na Ławce Skośnej Dodatniej",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 70,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 6.5,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.75,
                "klatka_srodek_mostek": 0.25,
                "bark_przod": 0.4
            }
        },
        "Wyciskanie_Gilotynowe_Skos_Dodatni": {
            "name": "Wyciskanie Gilotynowe na Skosie Dodatnim",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 85,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 7.0,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.8,
                "bark_przod": 0.6,
                "triceps_glowa_boczna_przysrodkowa": 0.3
            }
        },
        "Landmine_Press_Polsztanga": {
            "name": "Wyciskanie Półsztangi (Landmine Press) Stojąc / Klęcząc",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 35,
            "deep_extension_risk": "ZERO",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.7,
                "bark_przod": 0.5,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Pompki_Nogi_Podwyzszenie": {
            "name": "Pompki z Nogami na Podwyższeniu",
            "primary_target": "klatka_gora_obojczyk",
            "shoulder_abduction_angle": 50,
            "deep_extension_risk": "MEDIUM",
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "klatka_gora_obojczyk": 0.7,
                "klatka_srodek_mostek": 0.3,
                "bark_przod": 0.5,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },

        # =========================================================================
        # 2. CZĘŚĆ ŚRODKOWA / MOSTKOWO-ŻEBROWA (MIDDLE / STERNOCOSTAL)
        # =========================================================================
        "Wyciskanie_Hantli_Poziom": {
            "name": "Wyciskanie Hantli na Ławce Poziomej",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 55,
            "deep_extension_risk": "MEDIUM",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.7,
                "klatka_gora_obojczyk": 0.2,
                "klatka_dol_brzuszna": 0.1,
                "bark_przod": 0.4,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Wyciskanie_Sztangi_Poziom": {
            "name": "Wyciskanie Sztangi na Ławce Poziomej",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 80,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.75,
                "klatka_gora_obojczyk": 0.15,
                "klatka_dol_brzuszna": 0.1,
                "bark_przod": 0.6,
                "triceps_glowa_boczna_przysrodkowa": 0.5
            }
        },
        "Wyciskanie_Maszyna_Poziom": {
            "name": "Wyciskanie na Maszynie Poziomej",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 50,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.7,
                "klatka_gora_obojczyk": 0.2,
                "klatka_dol_brzuszna": 0.1,
                "bark_przod": 0.3,
                "triceps_glowa_boczna_przysrodkowa": 0.3
            }
        },
        "Wyciskanie_Smith_Poziom": {
            "name": "Wyciskanie na Maszynie Smitha na Ławce Poziomej",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 60,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.75,
                "klatka_gora_obojczyk": 0.15,
                "klatka_dol_brzuszna": 0.1,
                "bark_przod": 0.4,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Rozpietki_Kable_Poziomo": {
            "name": "Rozpiętki z Linkami Wyciągu Poziomo (Brama)",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 60,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.7,
                "klatka_gora_obojczyk": 0.2,
                "klatka_dol_brzuszna": 0.1,
                "bark_przod": 0.2
            }
        },
        "Rozpietki_Butterfly_Peck_Deck": {
            "name": "Rozpiętki na Maszynie Butterfly (Peck-Deck)",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 65,
            "deep_extension_risk": "MEDIUM",
            "base_hypertrophy_score": 9.6,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.7,
                "klatka_gora_obojczyk": 0.2,
                "klatka_dol_brzuszna": 0.1,
                "bark_przod": 0.2
            }
        },
        "Rozpietki_Hantle_Poziom": {
            "name": "Rozpiętki z Hantlami na Ławce Poziomej",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 70,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 6.5,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.75,
                "klatka_gora_obojczyk": 0.15,
                "klatka_dol_brzuszna": 0.1,
                "bark_przod": 0.3
            }
        },
        "Hex_Press_Squeeze_Press": {
            "name": "Wyciskanie Hantli Złączeniem (Squeeze Press / Hex Press)",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 30,
            "deep_extension_risk": "ZERO",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.8,
                "klatka_gora_obojczyk": 0.2,
                "triceps_glowa_boczna_przysrodkowa": 0.5
            }
        },
        "Pompki_Klasyczne": {
            "name": "Pompki Klasyczne (na Uchwytach / Płasko)",
            "primary_target": "klatka_srodek_mostek",
            "shoulder_abduction_angle": 55,
            "deep_extension_risk": "MEDIUM",
            "base_hypertrophy_score": 8.2,
            "muscle_contributions": {
                "klatka_srodek_mostek": 0.65,
                "klatka_gora_obojczyk": 0.2,
                "klatka_dol_brzuszna": 0.15,
                "bark_przod": 0.4,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },

        # =========================================================================
        # 3. CZĘŚĆ DOLNA / BRZUSZNA (LOWER / ABDOMINAL)
        # =========================================================================
        "Pompki_Na_Poreczach_Dips": {
            "name": "Pompki na Poręczach (Dips) w Nachyleniu do Przodu",
            "primary_target": "klatka_dol_brzuszna",
            "shoulder_abduction_angle": 35,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "klatka_dol_brzuszna": 0.7,
                "klatka_srodek_mostek": 0.3,
                "bark_przod": 0.6,
                "triceps_glowa_boczna_przysrodkowa": 0.5,
                "triceps_glowa_dluga": 0.3
            }
        },
        "Wyciskanie_Hantli_Skos_Ujemny": {
            "name": "Wyciskanie Hantli na Ławce Skośnej Ujemnej",
            "primary_target": "klatka_dol_brzuszna",
            "shoulder_abduction_angle": 45,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "klatka_dol_brzuszna": 0.75,
                "klatka_srodek_mostek": 0.25,
                "bark_przod": 0.2,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },
        "Wyciskanie_Sztangi_Skos_Ujemny": {
            "name": "Wyciskanie Sztangi na Ławce Skośnej Ujemnej",
            "primary_target": "klatka_dol_brzuszna",
            "shoulder_abduction_angle": 65,
            "deep_extension_risk": "MEDIUM",
            "base_hypertrophy_score": 7.5,
            "muscle_contributions": {
                "klatka_dol_brzuszna": 0.8,
                "klatka_srodek_mostek": 0.2,
                "bark_przod": 0.3,
                "triceps_glowa_boczna_przysrodkowa": 0.5
            }
        },
        "Wyciskanie_Maszyna_Skos_Ujemny": {
            "name": "Wyciskanie na Maszynie na Skos Ujemny",
            "primary_target": "klatka_dol_brzuszna",
            "shoulder_abduction_angle": 45,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "klatka_dol_brzuszna": 0.8,
                "klatka_srodek_mostek": 0.2,
                "bark_przod": 0.2,
                "triceps_glowa_boczna_przysrodkowa": 0.3
            }
        },
        "Rozpietki_Kable_Gora_Dol": {
            "name": "Rozpiętki z Linkami Wyciągu z Góry do Dołu",
            "primary_target": "klatka_dol_brzuszna",
            "shoulder_abduction_angle": 50,
            "deep_extension_risk": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "klatka_dol_brzuszna": 0.85,
                "klatka_srodek_mostek": 0.15,
                "bark_przod": 0.1
            }
        },
        "Rozpietki_Hantle_Skos_Ujemny": {
            "name": "Rozpiętki z Hantlami na Ławce Skośnej Ujemnej",
            "primary_target": "klatka_dol_brzuszna",
            "shoulder_abduction_angle": 65,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 6.5,
            "muscle_contributions": {
                "klatka_dol_brzuszna": 0.8,
                "klatka_srodek_mostek": 0.2,
                "bark_przod": 0.2
            }
        },
        "Przenoszenie_Hantla_Pullover": {
            "name": "Przenoszenie Hantla Za Głowę (Pullover)",
            "primary_target": "klatka_dol_brzuszna",
            "shoulder_abduction_angle": 30,
            "deep_extension_risk": "HIGH",
            "base_hypertrophy_score": 7.8,
            "muscle_contributions": {
                "klatka_dol_brzuszna": 0.5,
                "najszerszy_grzbietu": 0.4,
                "triceps_glowa_dluga": 0.3
            }
        }
    }

    results = {}

    for key, exercise in chest_database.items():
        # --- ETAP 1: FILTR BEZPIECZEŃSTWA (SAFETY GATE) ---
        is_safe = True
        disqualification_reason = ""

        if "kontuzja_barku" in injuries:
            if exercise["shoulder_abduction_angle"] > 60:
                is_safe = False
                disqualification_reason = "Kąt odwiedzenia ramienia > 60° (Niebezpieczny dla pierścienia rotatorów)."
            elif exercise["deep_extension_risk"] == "HIGH":
                is_safe = False
                disqualification_reason = "Wysokie ryzyko głębokiej ekstensji torebki stawowej."

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

        # Długie ramiona (reach_ratio > 1.2) zyskują na maszynach i wyciągach
        if reach_ratio > 1.2:
            if "Kable" in key or "Maszyna" in key or "Hammer" in key or "Butterfly" in key:
                score = min(10.0, score + 0.4)

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

    analysis = analyze_chest_exercises(test_biometrics, test_injuries)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))