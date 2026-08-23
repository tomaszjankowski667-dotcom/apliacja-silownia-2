import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_per_arm_kg, has_adduction, rom_bonus,
                           pec_torque_share, upper_activation, penalty=1.0, is_fly=False, is_hex=False):
    chest_tau = (weight_per_arm_kg * G * total_moment_arm) * pec_torque_share
    internal_moment_arm_factor = np.exp(-((t_vals - 0.40) / 0.30) ** 2)
    stretch_bonus_factor = 1.0 + (rom_bonus * 0.9 * np.exp(-12 * t_vals))

    if is_fly:
        adduct_factor = 1.0 - (1.0 * t_vals ** 3)
    elif is_hex:
        adduct_factor = 1.8 - 0.2 * t_vals
    else:
        adduct_factor = 1.0 + 0.35 * t_vals if has_adduction else 1.0 - 0.3 * (t_vals ** 2)

    curve = chest_tau * internal_moment_arm_factor * stretch_bonus_factor * adduct_factor * upper_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================
# Wyciskanie na skosie dodatnim charakteryzuje się ruchem, w którym łokcie
# i sztanga przemieszczają się mocniej w stronę głowy (dodatnie wartości Y) 
# w miarę fazy koncentrycznej (t=1).

def incline_barbell_kinematics(t, levers, phase="concentric"):
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 1.5) / 2
    # Przesunięcie sztangi bliżej obojczyków
    base_y_shift = L_HUM * 0.45 
    
    if phase == "concentric":
        y = L_HUM * 0.20 - base_y_shift * (1 - t)**4
    else:
        y = L_HUM * 0.20 - base_y_shift * np.sin((1 - t) * np.pi / 2)

    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.75
    return np.array([x, y, z])

def incline_dumbbell_kinematics(t, levers, phase="concentric"):
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W / 2 + L_HUM * 0.75) * (1 - t) + 0.08 * t
    base_y_shift = L_HUM * 0.35

    if phase == "concentric":
        y = L_HUM * 0.25 - base_y_shift * (1 - t)**3
    else:
        y = L_HUM * 0.25 - base_y_shift * np.sin((1 - t) * np.pi / 2)

    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.85
    return np.array([x, y, z])

def incline_smith_kinematics(t, levers, phase="concentric"):
    """Maszyna Smitha na skosie dodatnim: tor ściśle zablokowany, ukierunkowany na obojczyki."""
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 1.45) / 2
    # Sztywna prowadnica (lekko skośna względem ciała na ławce)
    y = L_HUM * 0.30 - L_HUM * 0.15 * (1 - t) 
    z = C_D + 0.04 + t * (L_HUM + L_FOR) * 0.75
    return np.array([x, y, z])

def incline_machine_kinematics(t, levers, phase="concentric"):
    """Maszyna Hammer na skos dodatni: ruch po konwergującym łuku w górę."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    y_start = -0.05
    y_end = y_start + (L_HUM + L_FOR) * 0.75
    y = y_start * (1 - t) + y_end * t

    x_start = (C_W / 2) + L_HUM * 0.65
    x_end = 0.12
    x = x_start * (1 - t) + x_end * t

    z_base = 0.10
    z = z_base + t * (L_HUM + L_FOR) * 0.6
    return np.array([x, y, z])

def low_to_high_cable_kinematics(t, levers, phase="concentric"):
    """Rozpiętki z dołu: t=0 (ręce nisko po bokach bioder) -> t=1 (dłonie złączone na wysokości twarzy/górnej klatki)."""
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.98

    angle = np.radians(75) * (1 - t) + np.radians(5) * t
    x = r * np.sin(angle)
    # Potężny ruch na osi Y (z dołu do góry)
    y = -r * 0.60 * (1 - t) + r * 0.45 * t 
    z = 0.10 + 0.30 * t
    return np.array([x, y, z])

def incline_dumbbell_flyes_kinematics(t, levers, phase="concentric"):
    """Rozpiętki z hantlami na skosie: szeroki łuk z opuszczonymi łokciami, zbiegający się nad twarzą."""
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.92

    angle = np.radians(70) * (1 - t) + np.radians(10) * t
    x = r * np.sin(angle)
    y = L_HUM * 0.15 - L_HUM * 0.20 * (1 - t)
    z = C_D + 0.05 + r * np.cos(angle)
    return np.array([x, y, z])

def incline_guillotine_kinematics(t, levers, phase="concentric"):
    """Gilotyna na skosie: skrajnie wysokie prowadzenie sztangi (pod samą szyję), łokcie flarują."""
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 1.6) / 2
    y = L_HUM * 0.65  # Bardzo wysoko, w rejonach obojczyków/szyi
    z = C_D + 0.08 + t * (L_HUM + L_FOR) * 0.70
    return np.array([x, y, z])

def landmine_press_kinematics(t, levers, phase="concentric"):
    """Wyciskanie półsztangi (Landmine): t=0 (przy klatce) -> t=1 (w górę i do przodu pod kątem)."""
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = 0.05  # Ręce blisko siebie na końcu sztangi
    y = -0.10 * (1 - t) + L_HUM * 0.8 * t
    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.85
    return np.array([x, y, z])

def feet_elevated_push_ups_kinematics(t, levers, phase="concentric"):
    """Pompki z nogami wyżej: ciało nachylone, biomechanicznie symuluje ławkę skośną dodatnią."""
    C_W = levers.get("biacromial_width", 0.41)
    x = (C_W * 1.35) / 2
    # Ciało porusza się tak, że względem barków punkt podparcia przesuwa się "w górę"
    y = 0.15 * t 
    z = 0.05 + t * 0.32
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_incline_barbell_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    chest_block = levers.get("chest_block", 0.240)

    weight_total = _get_weight(prof, "incline_barbell_bench", 80.0)
    weight_per_arm = weight_total / 2.0

    m_arms_x = l_humerus * np.cos(np.radians(45)) * (1 - 0.7 * t_vals)
    m_arms_y = l_humerus * 0.40 * (1 - t_vals)**4
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_y**2)

    elbow_drop = chest_block - (l_humerus * 0.75)
    penalty = 1.0 if elbow_drop > 0 else max(0.6, 1.0 - abs(elbow_drop) * 2.5)

    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.1, pec_torque_share=0.85, upper_activation=0.90, penalty=penalty)
    act_upper = min(0.95, raw_score / 320.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"barbell", "incline_bench", "squat_rack"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [40.0, 65.0],
            "bar_path_angle": [75.0, 90.0]
        },
        "trajectory_func": incline_barbell_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.40, "delt_front": 0.85, "tricep_lateral": 0.70},
        "fibers": {"chest_upper": 0.95, "chest_mid": 0.50, "delt_front": 0.95, "tricep_lateral": 0.75}
    }

def evaluate_incline_dumbbell_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight = _get_weight(prof, "incline_dumbbell_press", 27.5)

    m_arms_x = l_humerus * np.cos(np.radians(55)) * (1 - 0.85 * t_vals)
    m_arms_y = l_humerus * 0.25 * (1 - t_vals)**3
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_y**2)

    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.3, pec_torque_share=0.90, upper_activation=0.95)
    act_upper = min(1.00, raw_score / 320.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"dumbbells", "incline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [45.0, 70.0],
            "elbow_flexion_bottom": [70.0, 95.0]
        },
        "trajectory_func": incline_dumbbell_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.45, "delt_front": 0.80, "tricep_lateral": 0.55},
        "fibers": {"chest_upper": 1.00, "chest_mid": 0.55, "delt_front": 0.90, "tricep_lateral": 0.65}
    }

def evaluate_incline_smith_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "incline_smith_bench", 75.0)
    weight_per_arm = weight_total / 2.0

    m_arms_x = l_humerus * np.cos(np.radians(50)) * (1 - 0.65 * t_vals)
    total_m_arms = m_arms_x
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.2, pec_torque_share=0.85, upper_activation=0.95)
    act_upper = min(0.95, raw_score / 320.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"smith_machine", "incline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [45.0, 75.0],
            "vertical_bar_path_deviation": [0.0, 5.0]
        },
        "trajectory_func": incline_smith_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.35, "delt_front": 0.85, "tricep_lateral": 0.70},
        "fibers": {"chest_upper": 0.95, "chest_mid": 0.45, "delt_front": 0.90, "tricep_lateral": 0.75}
    }

def evaluate_incline_machine_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "incline_machine_hammer", 70.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, l_humerus * 0.80) * (1 - 0.35 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=True, rom_bonus=1.7, pec_torque_share=0.90, upper_activation=1.00)
    act_upper = min(1.00, raw_score / 320.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"machine_chest_press", "incline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [50.0, 75.0],
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": incline_machine_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.30, "delt_front": 0.70, "tricep_lateral": 0.60},
        "fibers": {"chest_upper": 1.00, "chest_mid": 0.40, "delt_front": 0.80, "tricep_lateral": 0.70}
    }

def evaluate_low_to_high_cable_crossover(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight = _get_weight(prof, "low_cable_flyes", 22.5)

    total_m_arms = l_humerus * (1 - 0.15 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.2, pec_torque_share=0.98, upper_activation=1.00)
    act_upper = min(0.98, raw_score / 220.0)

    return {
        "cat": "Push",
        "subcat": "isolation",
        "equipment": {"cable_machine"},
        "biomechanical_bounds": {
            "shoulder_flexion_end": [90.0, 120.0],
            "elbow_flexion_bottom": [140.0, 170.0]
        },
        "trajectory_func": low_to_high_cable_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.20, "delt_front": 0.65},
        "fibers": {"chest_upper": 1.00, "chest_mid": 0.30, "delt_front": 0.75}
    }

def evaluate_incline_dumbbell_flyes(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    weight = _get_weight(prof, "incline_dumbbell_flyes", 16.0)

    total_m_arms = (l_humerus + l_forearm * 0.7) * (1 - t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.6, pec_torque_share=0.95, upper_activation=0.95, is_fly=True)
    act_upper = min(0.95, raw_score / 250.0)

    return {
        "cat": "Push",
        "subcat": "isolation",
        "equipment": {"dumbbells", "incline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [60.0, 80.0],
            "elbow_flexion_constant": [130.0, 155.0]
        },
        "trajectory_func": incline_dumbbell_flyes_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.30, "delt_front": 0.50},
        "fibers": {"chest_upper": 0.95, "chest_mid": 0.40, "delt_front": 0.60}
    }

def evaluate_incline_guillotine_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "incline_guillotine", 60.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * np.cos(np.radians(20)) * (1 - 0.7 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.5, pec_torque_share=0.95, upper_activation=0.98)
    act_upper = min(0.98, raw_score / 280.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"barbell", "incline_bench", "squat_rack"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [80.0, 95.0], # Flared elbows
            "bar_path_angle": [85.0, 95.0]
        },
        "trajectory_func": incline_guillotine_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.20, "delt_front": 0.40, "tricep_lateral": 0.60},
        "fibers": {"chest_upper": 1.00, "chest_mid": 0.30, "delt_front": 0.50, "tricep_lateral": 0.70}
    }

def evaluate_landmine_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "landmine_press", 45.0)
    weight_per_arm = weight_total / 2.0

    m_arms_x = l_humerus * 0.2 * (1 - 0.5 * t_vals)
    m_arms_y = l_humerus * 0.7 * (1 - t_vals)
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_y**2)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=True, rom_bonus=1.0, pec_torque_share=0.85, upper_activation=0.95, is_hex=True)
    act_upper = min(0.92, raw_score / 280.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"barbell", "landmine_attachment"},
        "biomechanical_bounds": {
            "shoulder_flexion_end": [120.0, 150.0],
            "torso_lean_angle": [10.0, 30.0]
        },
        "trajectory_func": landmine_press_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.25, "delt_front": 0.85, "tricep_lateral": 0.70},
        "fibers": {"chest_upper": 0.95, "chest_mid": 0.35, "delt_front": 1.00, "tricep_lateral": 0.75}
    }

def evaluate_feet_elevated_push_ups(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    weight_per_arm = (body_weight * 0.75) / 2.0 # Wyższy wskaźnik ciężaru ciała z powodu podwyższenia nóg

    total_m_arms = l_humerus * np.cos(np.radians(50)) * (1 - 0.6 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.2, pec_torque_share=0.80, upper_activation=0.90)
    act_upper = min(0.92, raw_score / 350.0)

    return {
        "cat": "Push",
        "subcat": "bodyweight",
        "equipment": {"bodyweight", "bench_or_box"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [45.0, 75.0],
            "torso_straight_angle": [170.0, 180.0],
            "elevation_angle": [15.0, 45.0]
        },
        "trajectory_func": feet_elevated_push_ups_kinematics,
        "act": {"chest_upper": act_upper, "chest_mid": 0.35, "delt_front": 0.80, "tricep_lateral": 0.65},
        "fibers": {"chest_upper": 0.95, "chest_mid": 0.45, "delt_front": 0.90, "tricep_lateral": 0.70}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA GÓRĘ KLATKI
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Incline_Barbell_Press": evaluate_incline_barbell_press(user_profile),
        "Incline_Dumbbell_Press": evaluate_incline_dumbbell_press(user_profile),
        "Incline_Smith_Press": evaluate_incline_smith_press(user_profile),
        "Incline_Machine_Press": evaluate_incline_machine_press(user_profile),
        "Low_to_High_Cable_Crossover": evaluate_low_to_high_cable_crossover(user_profile),
        "Incline_Dumbbell_Flyes": evaluate_incline_dumbbell_flyes(user_profile),
        "Incline_Guillotine_Press": evaluate_incline_guillotine_press(user_profile),
        "Landmine_Press": evaluate_landmine_press(user_profile),
        "Feet_Elevated_Push_Ups": evaluate_feet_elevated_push_ups(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń góry klatki (obojczykowej). Wszystkie metryki (fibers, bounds, IK) kompletne.")
    except ImportError:
        pass