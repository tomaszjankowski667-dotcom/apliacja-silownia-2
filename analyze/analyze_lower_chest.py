import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_per_arm_kg, has_adduction, rom_bonus,
                           pec_torque_share, lower_activation, penalty=1.0, is_fly=False, is_pullover=False):
    chest_tau = (weight_per_arm_kg * G * total_moment_arm) * pec_torque_share
    
    if is_pullover:
        # W pulloverze największe ramię momentu jest w rozciągnięciu (t bliskie 0)
        internal_moment_arm_factor = np.exp(-((t_vals - 0.20) / 0.40) ** 2)
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-8 * t_vals))
    else:
        internal_moment_arm_factor = np.exp(-((t_vals - 0.40) / 0.30) ** 2)
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.9 * np.exp(-12 * t_vals))

    if is_fly:
        adduct_factor = 1.0 - (1.0 * t_vals ** 3)
    elif is_pullover:
        adduct_factor = 1.0 # Brak klasycznego przywodzenia horyzontalnego
    else:
        adduct_factor = 1.0 + 0.35 * t_vals if has_adduction else 1.0 - 0.3 * (t_vals ** 2)

    curve = chest_tau * internal_moment_arm_factor * stretch_bonus_factor * adduct_factor * lower_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================
# Wyciskanie na skosie ujemnym charakteryzuje się ruchem "w dół" tułowia
# (ujemne wartości Y), w stronę brzucha/bioder.

def chest_dips_kinematics(t, levers, phase="concentric"):
    """Pompki na poręczach (pochylone): ruch wzdłuż osi pionowej ciała, łokcie mocno ugięte na starcie."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    grip_x = (C_W * 1.35) / 2.0
    # Głębokie zejście do kąta ujemnego względem klatki, wypchnięcie prostujące ramię w dół
    y = -L_HUM * 0.20 - L_HUM * 0.60 * t
    z = 0.10 + t * (L_HUM + L_FOR) * 0.70
    return np.array([grip_x, y, z])

def decline_barbell_kinematics(t, levers, phase="concentric"):
    """Wyciskanie sztangi, skos ujemny: opuszczanie na dolny rejon klatki/splot słoneczny."""
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 1.5) / 2
    # Przesunięcie sztangi w stronę brzucha
    base_y_shift = L_HUM * 0.35 
    
    if phase == "concentric":
        y = -L_HUM * 0.25 - base_y_shift * t**2
    else:
        y = -L_HUM * 0.25 - base_y_shift * np.sin(t * np.pi / 2)

    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.70
    return np.array([x, y, z])

def decline_dumbbell_kinematics(t, levers, phase="concentric"):
    """Wyciskanie hantli, skos ujemny: mocna konwergencja nad dolną klatką."""
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W / 2 + L_HUM * 0.75) * (1 - t) + 0.08 * t
    base_y_shift = L_HUM * 0.30

    if phase == "concentric":
        y = -L_HUM * 0.20 - base_y_shift * t**2
    else:
        y = -L_HUM * 0.20 - base_y_shift * np.sin(t * np.pi / 2)

    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.75
    return np.array([x, y, z])

def decline_machine_kinematics(t, levers, phase="concentric"):
    """Maszyna na skos ujemny (np. Hammer): ruch po łuku wypychającym przed siebie i w dół."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    y_start = 0.10
    y_end = y_start - L_HUM * 0.80
    y = y_start * (1 - t) + y_end * t

    x_start = (C_W / 2) + L_HUM * 0.65
    x_end = 0.15
    x = x_start * (1 - t) + x_end * t

    z_base = 0.05
    z = z_base + t * (L_HUM + L_FOR) * 0.70
    return np.array([x, y, z])

def high_to_low_cable_kinematics(t, levers, phase="concentric"):
    """Rozpiętki wyciąg od góry do dołu (Brama): potężny wektor przywodzenia w dół przed biodra."""
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.96

    angle = np.radians(75) * (1 - t) + np.radians(5) * t
    x = r * np.sin(angle)
    # T z rąk szeroko i wysoko (Y dodatnie) schodzi przed biodra (Y ujemne)
    y = L_HUM * 0.50 * (1 - t) - L_HUM * 0.85 * t
    z = 0.25 - 0.15 * t
    return np.array([x, y, z])

def decline_dumbbell_flyes_kinematics(t, levers, phase="concentric"):
    """Rozpiętki hantlami na skosie ujemnym."""
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.94

    angle = np.radians(70) * (1 - t) + np.radians(10) * t
    x = r * np.sin(angle)
    y = -L_HUM * 0.25 - L_HUM * 0.15 * t
    z = C_D + 0.05 + r * np.cos(angle)
    return np.array([x, y, z])

def dumbbell_pullover_kinematics(t, levers, phase="concentric"):
    """Pullover hantlem za głowę: rotacja w płaszczyźnie strzałkowej (Y-Z), uderza w dolną klatkę i najszersze."""
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.88 # Lekko ugięte łokcie
    
    x = 0.05  # Dłonie złączone na uchwycie hantla
    # Kąt od ok. 170 stopni (za głową, mocno dodatnie Y, niskie Z) do 80 stopni (nad klatką/brzuchem)
    angle = np.radians(170) * (1 - t) + np.radians(80) * t
    
    y = r * np.cos(angle)
    z = C_D + r * np.sin(angle)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_chest_dips(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = prof.get("weight_kg", 85.0) + _get_weight(prof, "dips_added", 15.0)
    weight_per_arm = weight_total / 2.0

    # Dipy to potężny moment siły na ramieniu, silnie angażujący dolne włókna
    total_m_arms = l_humerus * np.cos(np.radians(25)) * (1 - 0.4 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.5, pec_torque_share=0.85, lower_activation=1.00)

    act_lower = min(1.00, raw_score / 360.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"dip_station"},
        "biomechanical_bounds": {
            "torso_forward_lean": [30.0, 50.0], # Nachylenie kluczowe dla zaangażowania klatki
            "elbow_flare_angle": [45.0, 65.0]
        },
        "trajectory_func": chest_dips_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": 0.60, "delt_front": 0.40, "tricep_lateral": 0.85},
        "fibers": {"chest_lower": 1.00, "chest_mid": 0.70, "delt_front": 0.50, "tricep_lateral": 0.90}
    }

def evaluate_decline_barbell_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    chest_block = levers.get("chest_block", 0.240)

    weight_total = _get_weight(prof, "decline_barbell_bench", 90.0)
    weight_per_arm = weight_total / 2.0

    m_arms_x = l_humerus * np.cos(np.radians(40)) * (1 - 0.65 * t_vals)
    m_arms_y = l_humerus * 0.35 * (1 - t_vals)**3
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_y**2)

    elbow_drop = chest_block - (l_humerus * 0.75)
    penalty = 1.0 if elbow_drop > 0 else max(0.65, 1.0 - abs(elbow_drop) * 2.0)

    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=0.9, pec_torque_share=0.85, lower_activation=0.95, penalty=penalty)
    act_lower = min(0.95, raw_score / 330.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"barbell", "decline_bench", "squat_rack"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [35.0, 60.0],
            "bar_path_angle": [90.0, 100.0]
        },
        "trajectory_func": decline_barbell_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": 0.50, "delt_front": 0.30, "tricep_lateral": 0.80},
        "fibers": {"chest_lower": 0.95, "chest_mid": 0.60, "delt_front": 0.40, "tricep_lateral": 0.85}
    }

def evaluate_decline_dumbbell_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight = _get_weight(prof, "decline_dumbbell_press", 30.0)

    m_arms_x = l_humerus * np.cos(np.radians(50)) * (1 - 0.80 * t_vals)
    m_arms_y = l_humerus * 0.20 * (1 - t_vals)**2
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_y**2)

    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.0, pec_torque_share=0.90, lower_activation=1.00)
    act_lower = min(1.00, raw_score / 330.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"dumbbells", "decline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [40.0, 65.0],
            "elbow_flexion_bottom": [75.0, 100.0]
        },
        "trajectory_func": decline_dumbbell_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": 0.55, "delt_front": 0.25, "tricep_lateral": 0.65},
        "fibers": {"chest_lower": 1.00, "chest_mid": 0.65, "delt_front": 0.35, "tricep_lateral": 0.70}
    }

def evaluate_decline_machine_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "decline_machine_hammer", 80.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, l_humerus * 0.85) * (1 - 0.40 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=True, rom_bonus=1.5, pec_torque_share=0.90, lower_activation=0.95)
    act_lower = min(0.95, raw_score / 330.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"machine_chest_press", "decline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [40.0, 65.0],
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": decline_machine_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": 0.50, "delt_front": 0.20, "tricep_lateral": 0.60},
        "fibers": {"chest_lower": 0.95, "chest_mid": 0.60, "delt_front": 0.30, "tricep_lateral": 0.70}
    }

def evaluate_high_to_low_cable_crossover(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight = _get_weight(prof, "high_cable_flyes", 25.0)

    total_m_arms = l_humerus * (1 - 0.15 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.0, pec_torque_share=0.98, lower_activation=1.00)
    act_lower = min(0.98, raw_score / 240.0)

    return {
        "cat": "Push",
        "subcat": "isolation",
        "equipment": {"cable_machine"},
        "biomechanical_bounds": {
            "shoulder_flexion_end": [10.0, 30.0], # Ręce kończą blisko bioder
            "elbow_flexion_bottom": [140.0, 170.0]
        },
        "trajectory_func": high_to_low_cable_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": 0.40, "delt_front": 0.15},
        "fibers": {"chest_lower": 1.00, "chest_mid": 0.50, "delt_front": 0.20}
    }

def evaluate_decline_dumbbell_flyes(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    weight = _get_weight(prof, "decline_dumbbell_flyes", 16.0)

    total_m_arms = (l_humerus + l_forearm * 0.7) * (1 - t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.4, pec_torque_share=0.95, lower_activation=0.95, is_fly=True)
    act_lower = min(0.95, raw_score / 260.0)

    return {
        "cat": "Push",
        "subcat": "isolation",
        "equipment": {"dumbbells", "decline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [55.0, 75.0],
            "elbow_flexion_constant": [130.0, 155.0]
        },
        "trajectory_func": decline_dumbbell_flyes_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": 0.35, "delt_front": 0.20},
        "fibers": {"chest_lower": 0.95, "chest_mid": 0.45, "delt_front": 0.30}
    }

def evaluate_dumbbell_pullover(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "dumbbell_pullover", 25.0)
    weight_per_arm = weight_total / 2.0

    # W pulloverze ramię momentu to długość całego ramienia, w rozciągnięciu (t=0) jest poziome
    total_m_arms = l_humerus * 1.5 * np.exp(-3 * t_vals) 
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=2.5, pec_torque_share=0.65, lower_activation=0.90, is_pullover=True)
    act_lower = min(0.90, raw_score / 250.0)

    return {
        "cat": "Pull/Push", # Pullover angażuje również plecy (najszerszy)
        "subcat": "compound",
        "equipment": {"dumbbell", "bench"},
        "biomechanical_bounds": {
            "shoulder_flexion_start": [160.0, 180.0],
            "elbow_flexion_constant": [140.0, 160.0]
        },
        "trajectory_func": dumbbell_pullover_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": 0.30, "lats": 0.85, "tricep_long": 0.50},
        "fibers": {"chest_lower": 0.95, "chest_mid": 0.40, "lats": 0.95, "tricep_long": 0.60}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA DÓŁ KLATKI
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Chest_Dips": evaluate_chest_dips(user_profile),
        "Decline_Barbell_Press": evaluate_decline_barbell_press(user_profile),
        "Decline_Dumbbell_Press": evaluate_decline_dumbbell_press(user_profile),
        "Decline_Machine_Press": evaluate_decline_machine_press(user_profile),
        "High_to_Low_Cable_Crossover": evaluate_high_to_low_cable_crossover(user_profile),
        "Decline_Dumbbell_Flyes": evaluate_decline_dumbbell_flyes(user_profile),
        "Dumbbell_Pullover": evaluate_dumbbell_pullover(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń dołu klatki (brzusznej). Wszystkie metryki (fibers, bounds, IK) kompletne.")
    except ImportError:
        pass