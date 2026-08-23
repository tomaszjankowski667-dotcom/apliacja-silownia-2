"""
ARMS MODULE (arms.py)
---------------------
Pełny atlas ćwiczeń dla ramion i przedramion z podziałem na aktony:
1. BICEPS: głowa długa (biceps_glowa_dluga), głowa krótka (biceps_glowa_krotka)
2. RAMIENNY / RAMIENNO-PROMIENIOWY: (ramienny_ramienno_promieniowy)
3. TRICEPS: głowa długa (triceps_glowa_dluga), głowa boczna i przyśrodkowa (triceps_glowa_boczna_przysrodkowa)
4. PRZEDRAMIONA: prostowniki (przedramie_prostowniki), zginacze (przedramie_zginacze), chwyt (sila_chwytu_przedramie)
"""


def analyze_arms_exercises(biometrics: dict, injuries: list) -> dict:
    reach_ratio = biometrics.get("reach_ratio", 1.0)

    arms_database = {
        # =========================================================================
        # 1. BICEPS - GŁOWA DŁUGA (LONG HEAD - ŁOKIEĆ ZA TUŁOWIEM / WĄSKI CHWYT)
        # =========================================================================
        "Uginanie_Hantle_Lawka_Skosna": {
            "name": "Uginanie Ramion z Hantlami na Ławce Skośnej z Oparciem",
            "primary_target": "biceps_glowa_dluga",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.8,  # Maksymalne rozciągnięcie w biodrze/barku
            "muscle_contributions": {
                "biceps_glowa_dluga": 0.75,
                "biceps_glowa_krotka": 0.25,
                "ramienny_ramienno_promieniowy": 0.2
            }
        },
        "Uginanie_Kable_Tylem_Wyciag": {
            "name": "Uginanie Ramion z Linkami Wyciągu Dolnego Stojąc (Tyłem do Wyciągu)",
            "primary_target": "biceps_glowa_dluga",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.6,
            "muscle_contributions": {
                "biceps_glowa_dluga": 0.8,
                "biceps_glowa_krotka": 0.2,
                "ramienny_ramienno_promieniowy": 0.15
            }
        },
        "Uginanie_Sztanga_Waski_Chwyt": {
            "name": "Uginanie Ramion Sztangą z Wąskim Chwytem",
            "primary_target": "biceps_glowa_dluga",
            "elbow_stress": "MEDIUM",
            "wrist_stress": "HIGH",  # Duża rotacja w nadgarstkach przy prostej sztandze
            "base_hypertrophy_score": 8.0,
            "muscle_contributions": {
                "biceps_glowa_dluga": 0.7,
                "biceps_glowa_krotka": 0.3,
                "ramienny_ramienno_promieniowy": 0.2
            }
        },
        "Drag_Curl_Sztanga": {
            "name": "Drag Curl (Uginanie Ramion ze Sztangą Prowadzoną wzdłuż Tułowia)",
            "primary_target": "biceps_glowa_dluga",
            "elbow_stress": "LOW",
            "wrist_stress": "MEDIUM",
            "base_hypertrophy_score": 8.5,
            "muscle_contributions": {
                "biceps_glowa_dluga": 0.75,
                "biceps_glowa_krotka": 0.25,
                "bark_tyl": 0.2
            }
        },

        # =========================================================================
        # 1. BICEPS - GŁOWA KRÓTKA (SHORT HEAD - ŁOKIEĆ PRZED TUŁOWIEM / SZEROKI CHWYT)
        # =========================================================================
        "Modlitewnik_Sztanga_Lamana": {
            "name": "Uginanie Ramion na Modlitewniku ze Sztangą Łamaną (Preacher Curl)",
            "primary_target": "biceps_glowa_krotka",
            "elbow_stress": "HIGH",  # Duże obciążenie w pełnym wyproście łokcia
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "biceps_glowa_krotka": 0.7,
                "biceps_glowa_dluga": 0.3,
                "ramienny_ramienno_promieniowy": 0.3
            }
        },
        "Modlitewnik_Hantel_Maszyna": {
            "name": "Uginanie Ramion na Modlitewniku z Hantlem / na Maszynie",
            "primary_target": "biceps_glowa_krotka",
            "elbow_stress": "MEDIUM",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "biceps_glowa_krotka": 0.75,
                "biceps_glowa_dluga": 0.25,
                "ramienny_ramienno_promieniowy": 0.25
            }
        },
        "Uginanie_Sztanga_Szeroki_Chwyt": {
            "name": "Uginanie Ramion ze Sztangą z Szerokim Chwytem",
            "primary_target": "biceps_glowa_krotka",
            "elbow_stress": "LOW",
            "wrist_stress": "MEDIUM",
            "base_hypertrophy_score": 8.2,
            "muscle_contributions": {
                "biceps_glowa_krotka": 0.7,
                "biceps_glowa_dluga": 0.3
            }
        },
        "Uginanie_Kable_Brama_Kulturystyczne": {
            "name": "Uginanie Ramion z Linkami Wyciągu Górnego Bramy (Podwójny Biceps)",
            "primary_target": "biceps_glowa_krotka",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "biceps_glowa_krotka": 0.8,
                "biceps_glowa_dluga": 0.2
            }
        },
        "Uginanie_Skoncentrowane_Hantel": {
            "name": "Uginanie Skoncentrowane z Hantlem w Podparciu o Udo",
            "primary_target": "biceps_glowa_krotka",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "biceps_glowa_krotka": 0.75,
                "biceps_glowa_dluga": 0.25,
                "ramienny_ramienno_promieniowy": 0.2
            }
        },

        # =========================================================================
        # 1. MIĘŚNIEŃ RAMIENNY I RAMIENNO-PROMIENIOWY (BRACHIALIS & BRACHIORADIALIS)
        # =========================================================================
        "Uginanie_Mlotkowe_Hantlis": {
            "name": "Uginanie Ramion z Hantlami w Chwycie Młotkowym (Hammer Curl)",
            "primary_target": "ramienny_ramienno_promieniowy",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "ramienny_ramienno_promieniowy": 0.8,
                "biceps_glowa_dluga": 0.3,
                "przedramie_prostowniki": 0.3
            }
        },
        "Uginanie_Mlotkowe_Kable_Lina": {
            "name": "Uginanie Ramion na Wyciągu z Liną (Cable Hammer Curl)",
            "primary_target": "ramienny_ramienno_promieniowy",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "ramienny_ramienno_promieniowy": 0.85,
                "biceps_glowa_dluga": 0.25,
                "przedramie_prostowniki": 0.2
            }
        },
        "Uginanie_Nachwytem_Reverse_Curl": {
            "name": "Uginanie Ramion ze Sztangą Nachwytem (Reverse Curl)",
            "primary_target": "ramienny_ramienno_promieniowy",
            "elbow_stress": "MEDIUM",
            "wrist_stress": "MEDIUM",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "ramienny_ramienno_promieniowy": 0.7,
                "przedramie_prostowniki": 0.6,
                "biceps_glowa_dluga": 0.2
            }
        },

        # =========================================================================
        # 2. TRICEPS - GŁOWA DŁUGA (LONG HEAD - RUCHY OVERHEAD / ZA GŁOWĘ)
        # =========================================================================
        "Prostowanie_Nad_Glowa_Hantel": {
            "name": "Prostowanie Ramion z Hantlem Nad Głową",
            "primary_target": "triceps_glowa_dluga",
            "elbow_stress": "MEDIUM",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "triceps_glowa_dluga": 0.8,
                "triceps_glowa_boczna_przysrodkowa": 0.2
            }
        },
        "Francuskie_Wyciskanie_Czol_Za_Glowe": {
            "name": "Francuskie Wyciskanie Sztangi Łamanej do Czoła / za Głowę",
            "primary_target": "triceps_glowa_dluga",
            "elbow_stress": "HIGH",  # Wysoki stres na przyczep kątowy łokcia
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "triceps_glowa_dluga": 0.7,
                "triceps_glowa_boczna_przysrodkowa": 0.3
            }
        },
        "Prostowanie_Nad_Glowa_Kable_Lina": {
            "name": "Prostowanie Ramion z Liną Wyciągu Nad Głową",
            "primary_target": "triceps_glowa_dluga",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.8,  # Stałe napięcie w pełnym rozciągnięciu
            "muscle_contributions": {
                "triceps_glowa_dluga": 0.85,
                "triceps_glowa_boczna_przysrodkowa": 0.25
            }
        },
        "Kickback_Hantel_Opad": {
            "name": "Prostowanie Ramienia z Hantlem w Opadzie Tułowia (Kickback)",
            "primary_target": "triceps_glowa_dluga",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 7.0,  # Brak oporu na dole ruchu
            "muscle_contributions": {
                "triceps_glowa_dluga": 0.6,
                "triceps_glowa_boczna_przysrodkowa": 0.4
            }
        },

        # =========================================================================
        # 2. TRICEPS - GŁOWA BOCZNA I PRZYŚRODKOWA (LATERAL & MEDIAL HEAD)
        # =========================================================================
        "Pushdown_Wyciag_Drazek": {
            "name": "Prostowanie Ramion na Wyciągu Górnym z Drążkiem Prostym / Łamanym",
            "primary_target": "triceps_glowa_boczna_przysrodkowa",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "triceps_glowa_boczna_przysrodkowa": 0.8,
                "triceps_glowa_dluga": 0.2
            }
        },
        "Pushdown_Wyciag_Lina": {
            "name": "Prostowanie Ramion na Wyciągu Górnym z Liną (Cable Rope Pushdown)",
            "primary_target": "triceps_glowa_boczna_przysrodkowa",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "triceps_glowa_boczna_przysrodkowa": 0.85,
                "triceps_glowa_dluga": 0.2
            }
        },
        "Wyciskanie_Wasko_Poziom": {
            "name": "Wyciskanie Sztangi w Leżeniu w Wąskim Chwycie (Close-Grip Bench Press)",
            "primary_target": "triceps_glowa_boczna_przysrodkowa",
            "elbow_stress": "MEDIUM",
            "wrist_stress": "MEDIUM",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "triceps_glowa_boczna_przysrodkowa": 0.7,
                "klatka_srodek_mostek": 0.4,
                "bark_przod": 0.4,
                "triceps_glowa_dluga": 0.2
            }
        },
        "Dips_Triceps_Pionowo": {
            "name": "Pompki na Poręczach z Pionowym Ustawieniem Tułowia (Triceps Dips)",
            "primary_target": "triceps_glowa_boczna_przysrodkowa",
            "elbow_stress": "MEDIUM",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 8.8,
            "muscle_contributions": {
                "triceps_glowa_boczna_przysrodkowa": 0.7,
                "triceps_glowa_dluga": 0.3,
                "klatka_dol_brzuszna": 0.3,
                "bark_przod": 0.4
            }
        },
        "Pompki_Odwrotne_Lawka": {
            "name": "Pompki Odwrotne w Podparciu o Ławkę (Bench Dips)",
            "primary_target": "triceps_glowa_boczna_przysrodkowa",
            "elbow_stress": "HIGH",
            "wrist_stress": "HIGH",
            "base_hypertrophy_score": 7.5,  # Ryzykowne dla torebki stawowej barku
            "muscle_contributions": {
                "triceps_glowa_boczna_przysrodkowa": 0.75,
                "triceps_glowa_dluga": 0.25,
                "bark_przod": 0.5
            }
        },
        "Prostowanie_Jednoracz_Wyciag": {
            "name": "Prostowanie Ramion Jednorącz Nachwytem / Podchwytem na Wyciągu",
            "primary_target": "triceps_glowa_boczna_przysrodkowa",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "triceps_glowa_boczna_przysrodkowa": 0.85,
                "triceps_glowa_dluga": 0.15
            }
        },

        # =========================================================================
        # 3. PRZEDRAMIONA - PROSTOWNIKI (EXTENSORS)
        # =========================================================================
        "Prostowanie_Nadgarstkow_Nachwyt_Lawka": {
            "name": "Prostowanie Nadgarstków ze Sztangą / Hantlami w Nachwycie na Ławce",
            "primary_target": "przedramie_prostowniki",
            "elbow_stress": "LOW",
            "wrist_stress": "MEDIUM",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "przedramie_prostowniki": 0.95
            }
        },
        "Prostowanie_Nadgarstkow_Kable_Dol": {
            "name": "Prostowanie Nadgarstków z Linką Wyciągu Dolnego",
            "primary_target": "przedramie_prostowniki",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.5,
            "muscle_contributions": {
                "przedramie_prostowniki": 0.95
            }
        },

        # =========================================================================
        # 3. PRZEDRAMIONA - ZGINACZE (FLEXORS)
        # =========================================================================
        "Uginanie_Nadgarstkow_Podchwyt_Lawka": {
            "name": "Uginanie Nadgarstków ze Sztangą / Hantlami w Podchwycie na Ławce",
            "primary_target": "przedramie_zginacze",
            "elbow_stress": "LOW",
            "wrist_stress": "MEDIUM",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "przedramie_zginacze": 0.95
            }
        },
        "Uginanie_Nadgarstkow_Tyl_Plecow": {
            "name": "Uginanie Nadgarstków ze Sztangą z Tyłu Pleców Stojąc",
            "primary_target": "przedramie_zginacze",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "przedramie_zginacze": 0.95
            }
        },

        # =========================================================================
        # 3. SIŁA CHWYTU I MIĘŚNIE WSPOMAGAJĄCE (GRIP & AUXILIARY)
        # =========================================================================
        "Nawijanie_Ciezarka_Wrist_Roller": {
            "name": "Nawijanie Ciężarka na Drążek na Sznurku (Wrist Roller)",
            "primary_target": "sila_chwytu_przedramie",
            "elbow_stress": "LOW",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.8,
            "muscle_contributions": {
                "przedramie_prostowniki": 0.5,
                "przedramie_zginacze": 0.5,
                "sila_chwytu_przedramie": 0.8
            }
        },
        "Sciskacze_Dloni": {
            "name": "Ściskanie Ściskaczy Dłoni z Progresywnym Oporem (Hand Grippers)",
            "primary_target": "sila_chwytu_przedramie",
            "elbow_stress": "ZERO",
            "wrist_stress": "ZERO",
            "base_hypertrophy_score": 9.0,
            "muscle_contributions": {
                "sila_chwytu_przedramie": 0.95,
                "przedramie_zginacze": 0.4
            }
        },
        "Spacer_Farmera_Szczypcowy_Pinch_Grip": {
            "name": "Spacer Farmera w Chwycie Szczypcowym (Pinch Grip Farmer's Walk)",
            "primary_target": "sila_chwytu_przedramie",
            "elbow_stress": "ZERO",
            "wrist_stress": "LOW",
            "base_hypertrophy_score": 9.2,
            "muscle_contributions": {
                "sila_chwytu_przedramie": 0.9,
                "przedramie_zginacze": 0.5,
                "czworoboczny_gora": 0.3
            }
        }
    }

    results = {}

    for key, exercise in arms_database.items():
        # --- ETAP 1: FILTR BEZPIECZEŃSTWA (SAFETY GATE) ---
        is_safe = True
        disqualification_reason = ""

        # Kontuzja łokcia (np. łokieć tenisisty / golfisty / bolesność przyczepów)
        if "bol_lokcia" in injuries or "lokiec_tenisisty" in injuries or "lokiec_golfisty" in injuries:
            if exercise["elbow_stress"] == "HIGH":
                is_safe = False
                disqualification_reason = "Wysoki stres rozciągający na przyczepy stawu łokciowego."

        # Kontuzja nadgarstka
        if "bol_nadgarstka" in injuries:
            if exercise["wrist_stress"] == "HIGH":
                is_safe = False
                disqualification_reason = "Przekroczony bezpieczny próg obciążenia nadgarstków."

        # Kontuzja barku (wyklucza niebezpieczne Bench Dips i wymuszone skrajne wyprosty)
        if "kontuzja_barku" in injuries and key == "Pompki_Odwrotne_Lawka":
            is_safe = False
            disqualification_reason = "Niebezpieczna głęboka ekstensja w stawie ramiennym."

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

        # Długie ramiona (reach_ratio > 1.2) zyskują na ćwiczeniach z wyciągami (stałe napięcie)
        if reach_ratio > 1.2:
            if "Kable" in key or "Wyciag" in key or "Maszyn" in key:
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
    test_injuries = ["lokiec_tenisisty"]

    analysis = analyze_arms_exercises(test_biometrics, test_injuries)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))