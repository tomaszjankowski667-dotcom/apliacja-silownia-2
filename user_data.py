"""
USER DATA (user_data.py) - SINGLE SOURCE OF TRUTH
-------------------------------------------------
Główna baza profili użytkowników. Z tego pliku czerpią absolutnie
wszystkie mikroserwisy (Biomechanika, Planer, CV, Bio-Score).
"""

PROFILES = {
    "Brat (Z_Zdjecia)": {
        # --- 1. PODSTAWOWE DANE FIZYCZNE I REGENERACJA ---
        "height_cm": 176.0,
        "weight_kg": 87.0,
        "bf_percent": 14.0,           # Wymagane przez Bio-Score
        "wrist_cm": 18.0,             # Wymagane do oceny ramy kostnej
        "ankle_cm": 23.0,             # Wymagane do oceny ramy kostnej
        "years_of_training": 2.5,     # Wymagane przez Planer (Model McDonalda)
        "recovery_factor": 1.0,       # 1.0 = normalnie, <1.0 = niedobór snu/diety

        # --- 2. DŹWIGNIE BIOMECHANICZNE (W METRACH) ---
        # Dostarczane bezpośrednio przez cv_body_segmentation.py
        "levers": {
            "L_humerus": 0.326,       # Ramię (dawniej L_h)
            "L_forearm": 0.285,       # Przedramię
            "L_femur": 0.449,         # Kość udowa
            "L_tibia": 0.410,         # Piszczel
            "L_torso": 0.480,         # Tułów (od środka barków do bioder)
            "biacromial_width": 0.410,# Szerokość barków
            "chest_block": 0.240      # Głębokość klatki (blokada dla wyciskania)
        },

        # --- 3. WAGI ROBOCZE (MAX 8-10 REP) ---
        "weights": {
            "dumbbell_press": 40.0,
            "barbell_bench": 110.0,
            "squat": 130.0,
            "deadlift_rdl": 145.0,
            "pullup_pulldown": 85.0,
            "machine_hammer": 100.0,
            "smith_machine": 120.0,
            "cable_flyes": 40.0,
            "pushup": 87.0 * 0.65     # Masa ciała * współczynnik oporu
        },

        # --- 4. HISTORYCZNE ZAANGAŻOWANIE AKTONÓW (0.0 - 1.0) ---
        # Wyliczane przez fatigue_volume_tracker.py.
        # Niskie wartości (< 0.3) uruchamiają mnożnik uśpionych włókien (x12).
        "historical_akton_engagement": {
            "chest_upper": 0.30, "chest_mid": 0.95, "chest_lower": 0.15,
            "delt_front": 0.95, "delt_side": 0.40, "delt_rear": 0.20,
            "tricep_long": 0.30, "tricep_lateral": 0.85,
            "bicep_long": 0.80, "bicep_short": 0.50,
            "abs_upper": 0.80, "abs_lower": 0.20, "abs_oblique": 0.10,
            "quad_vastus": 0.90, "quad_rectus": 0.50,
            "hamstring": 0.40, "calves": 0.10,
            "back_lats": 0.85, "back_mid": 0.40, "back_lower": 0.60
        },

        # --- 5. DOMYŚLNY SPRZĘT UŻYTKOWNIKA ---
        # Filtrowane przez smart_equipment_manager.py
        "available_equipment": [
            "barbell", "dumbbells", "bench", "squat_rack",
            "cable_machine", "pullup_bar", "leg_press_machine"
        ]
    }
}