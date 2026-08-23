import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_per_arm_kg, rom_bonus,
                           rhomboid_torque_share, rhomboid_activation, penalty=1.0, 
                           is_wide_row=False, is_pause_rep=False, is_fly=False):
    """
    Krzywa oporu dla mięśni równoległobocznych.
    Najważniejszym elementem jest faza końcowa (t=1), w której następuje pełna retrakcja łopatki.
    """
    tau = (weight_per_arm_kg * G * total_moment_arm) * rhomboid_torque_share
    
    if is_pause_rep:
        # Pauza izometryczna w szczycie ruchu: Ekstremalny mnożnik napięcia na samej górze (t zbliżone do 1)
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.5 * np.exp(-5 * t_vals)) # Słabszy stretch na dole
        # Napięcie szybuje w górę przy dociągnięciu łopatek do kręgosłupa
        leverage_factor = 0.5 + 0.8 * np.exp(4 * (t_vals - 1)) 
    elif is_wide_row:
        # Wiosłowania szeroko na maszynie: Płynny wzrost zaangażowania środka pleców
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.8 * np.exp(-8 * t_vals))
        leverage_factor = 0.7 + 0.5 * t_vals 
    elif is_fly:
        # Odwrotne rozpiętki z akcentem na łopatki: Stałe napięcie z wyciągu/maszyny
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-6 * t_vals))
        leverage_factor = 0.8 + 0.4 * t_vals # Najciężej dopiąć na końcu
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * rhomboid_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def machine_wide_row_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie na maszynie (podparcie klatki, chwyt szeroki)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 1.6) / 2 # Szeroki chwyt zdejmuje pracę z latsów
    # Maszyna izoluje tułów. Ruch rąk do tyłu.
    y = -L_HUM * 0.20 # Minimalny ruch w pionie (jeśli maszyna jest horyzontalna)
    z = (L_HUM) * 1.1 * (1 - t) - L_HUM * 0.20 * t # Z przodu do mocnego cofnięcia łokci
    return np.array([x, y, z])

def dumbbell_row_pause_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie hantlami w opadzie tułowia z PAUZĄ (Rhomboid Focus)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 1.2) / 2 # Łokcie szerzej niż przy tułowiu, ale swobodnie
    # Tułów opuszczony, ciąg pionowo w górę przeciw grawitacji
    y = -L_HUM * 1.2 * (1 - t)
    z = -L_HUM * 0.25 * t # Cofnięcie łokci za linię pleców w szczytowym momencie
    return np.array([x, y, z])

def reverse_fly_scapular_kinematics(t, levers, phase="concentric"):
    """Odwrotne rozpiętki z potężnym akcentem na retrakcję łopatek."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.90 # Lekko ugięte łokcie
    
    # Kąt do 0 stopni (wzdłuż osi barków), ale fizycznie użytkownik stara się 
    # wejść na ujemne stopnie, żeby dopiąć łopatki z tyłu.
    angle = np.radians(90) * (1 - t) - np.radians(10) * t 
    
    x = (C_W / 2) + r * np.cos(angle)
    y = 0.0
    z = r * np.sin(angle)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_machine_wide_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight_total = _get_weight(prof, "machine_wide_row", 60.0)
    weight_per_arm = weight_total / 2.0

    # Dźwignia skrócona do okolic łokcia. Maszyna gwarantuje stabilność tułowia.
    total_m_arms = np.full_like(t_vals, l_humerus * 0.85)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.2, rhomboid_torque_share=0.85, rhomboid_activation=0.95, is_wide_row=True)
    act_rhomboid = min(0.95, raw_score / 200.0)

    return {
        "cat": "Back",
        "subcat": "isolation",
        "equipment": {"rowing_machine", "chest_pad"},
        "biomechanical_bounds": {
            "elbow_flare_angle": [70.0, 90.0], # Klucz: łokcie wysoko
            "torso_stability": [98.0, 100.0]
        },
        "trajectory_func": machine_wide_row_kinematics,
        "act": {"rhomboids": act_rhomboid, "traps_mid": 0.85, "delt_rear": 0.70},
        "fibers": {"rhomboid_major": 1.00, "rhomboid_minor": 0.90, "traps_mid": 0.80}
    }

def evaluate_dumbbell_row_pause(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight = _get_weight(prof, "dumbbell_row_pause", 25.0) # Celowo mniejszy ciężar ze względu na pauzę izometryczną

    total_m_arms = l_humerus * np.cos(np.radians(20)) * (0.5 + 0.5 * np.sin(t_vals * np.pi / 2))
    
    # Dodano mnożnik dla pauzy w szczytowym momencie (is_pause_rep=True) - genialne pobudzenie włókien wolnokurczliwych łopatki
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.5, rhomboid_torque_share=0.95, rhomboid_activation=1.00, is_pause_rep=True)
    act_rhomboid = min(1.00, raw_score / 180.0)

    return {
        "cat": "Back",
        "subcat": "compound_isolation",
        "equipment": {"dumbbells", "bench"},
        "biomechanical_bounds": {
            "scapular_retraction_duration": [1.0, 3.0], # Czas trzymania pauzy w sekundach
            "torso_forward_lean": [70.0, 90.0]
        },
        "trajectory_func": dumbbell_row_pause_kinematics,
        "act": {"rhomboids": act_rhomboid, "traps_mid": 0.90, "lats": 0.30},
        "fibers": {"rhomboid_major": 1.00, "rhomboid_minor": 1.00, "traps_mid": 0.95}
    }

def evaluate_reverse_fly_scapular(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "reverse_fly_scapular", 20.0)

    # Ramię momentu rośnie wraz z odwodzeniem, a przy próbie "złamania" osi barków naprężenie idzie w kosmos
    total_m_arms = (l_humerus + l_forearm) * 0.85 * (0.6 + 0.4 * np.sin(t_vals * np.pi / 2))
    
    # Skupienie neuromięśniowe na zbliżeniu łopatek, a nie tylko na ruchu barku
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.5, rhomboid_torque_share=0.85, rhomboid_activation=0.98, is_fly=True)
    act_rhomboid = min(0.98, raw_score / 160.0)

    return {
        "cat": "Back",
        "subcat": "isolation",
        "equipment": {"cable_machine", "pec_deck_machine"},
        "biomechanical_bounds": {
            "shoulder_horizontal_abduction": [90.0, 105.0], # Wymuszenie przekroczenia linii prostej barków (hiper-odwodzenie horyzontalne)
            "scapular_retraction_end": [95.0, 100.0]
        },
        "trajectory_func": reverse_fly_scapular_kinematics,
        "act": {"rhomboids": act_rhomboid, "delt_rear": 0.80, "traps_mid": 0.85},
        "fibers": {"rhomboid_major": 1.00, "rhomboid_minor": 0.90, "delt_rear": 0.85}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA MIĘŚNIE RÓWNOLEGŁOBOCZNE
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Machine_Wide_Row": evaluate_machine_wide_row(user_profile),
        "Dumbbell_Row_Pause": evaluate_dumbbell_row_pause(user_profile),
        "Reverse_Fly_Scapular_Focus": evaluate_reverse_fly_scapular(user_profile)
    }

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from user_data import PROFILES
        test_prof = PROFILES.get("Brat (Z_Zdjecia)")
        if test_prof:
            data = get_exercises_data(test_prof)
            print(f"Załadowano {len(data)} ćwiczeń na mięśnie równoległoboczne (Rhomboids). Wszystkie metryki kompletne.")
    except ImportError:
        pass