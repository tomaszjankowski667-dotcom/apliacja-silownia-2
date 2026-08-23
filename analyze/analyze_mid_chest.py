import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_per_arm_kg, has_adduction, rom_bonus,
                           pec_torque_share, mid_activation, penalty=1.0, is_fly=False, is_hex=False):
    chest_tau = (weight_per_arm_kg * G * total_moment_arm) * pec_torque_share
    internal_moment_arm_factor = np.exp(-((t_vals - 0.40) / 0.30) ** 2)
    stretch_bonus_factor = 1.0 + (rom_bonus * 0.9 * np.exp(-12 * t_vals))

    if is_fly:
        adduct_factor = 1.0 - (1.0 * t_vals ** 3)
    elif is_hex:
        adduct_factor = 1.8 - 0.2 * t_vals
    else:
        adduct_factor = 1.0 + 0.35 * t_vals if has_adduction else 1.0 - 0.3 * (t_vals ** 2)

    curve = chest_tau * internal_moment_arm_factor * stretch_bonus_factor * adduct_factor * mid_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def barbell_kinematics(t, levers, phase="concentric"):
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 1.5) / 2
    base_y_shift = L_HUM * 0.30

    if phase == "concentric":
        y = -base_y_shift * (1 - t)**4
    else:
        y = -base_y_shift * np.sin((1 - t) * np.pi / 2)

    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.7
    return np.array([x, y, z])

def dumbbell_kinematics(t, levers, phase="concentric"):
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W / 2 + L_HUM * 0.8) * (1 - t) + 0.05 * t
    base_y_shift = L_HUM * 0.15

    if phase == "concentric":
        y = -base_y_shift * (1 - t)**3
    else:
        y = -base_y_shift * np.sin((1 - t) * np.pi / 2)

    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.8
    return np.array([x, y, z])

def machine_kinematics(t, levers, phase="concentric"):
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    y_start = 0.15
    y_end = y_start + (L_HUM + L_FOR) * 0.9
    y = y_start * (1 - t) + y_end * t

    x_start = (C_W / 2) + L_HUM * 0.7
    x_end = 0.15
    x = x_start * (1 - t) + x_end * t

    z_base = 0.05
    z = z_base + 0.05 * np.sin(t * np.pi)
    return np.array([x, y, z])

def smith_press_kinematics(t, levers, phase="concentric"):
    """Maszyna Smitha: tor ściśle zablokowany w pionie z lekkim przesunięciem na mostek."""
    C_W = levers.get("biacromial_width", 0.41)
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 1.45) / 2
    y = -L_HUM * 0.20  # Stała linia prowadnicy
    z = C_D + 0.04 + t * (L_HUM + L_FOR) * 0.75
    return np.array([x, y, z])

def cable_kinematics(t, levers, phase="concentric"):
    """Brama: t=0 (szeroko po bokach) -> t=1 (dłonie zbiegają się ku przodowi przed klatkę)."""
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.96

    angle = np.radians(68) * (1 - t) + np.radians(5) * t
    x = r * np.sin(angle)
    y = r * np.cos(angle)
    z = 0.20 - 0.04 * t
    return np.array([x, y, z])

def butterfly_pec_deck_kinematics(t, levers, phase="concentric"):
    """Butterfly / Pec-Deck: t=0 (odwiedzenie po bokach) -> t=1 (dopięcie z przodu przed mostkiem)."""
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.94

    angle = np.radians(70) * (1 - t) + np.radians(5) * t
    x = r * np.sin(angle)
    y = r * np.cos(angle)
    z = 0.15
    return np.array([x, y, z])

def dumbbell_flyes_kinematics(t, levers, phase="concentric"):
    """Rozpiętki z hantlami leżąc: ruch po szerokim łuku z lekkim ugięciem w łokciu."""
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.94

    angle = np.radians(68) * (1 - t) + np.radians(8) * t
    x = r * np.sin(angle)
    y = -0.05 * (1 - t)
    z = C_D + 0.05 + r * np.cos(angle)
    return np.array([x, y, z])

def hex_press_kinematics(t, levers, phase="concentric"):
    """Hex / Squeeze Press: hantle dociśnięte do siebie przez cały zakres ruchu (x bliskie 0)."""
    C_D = levers.get("chest_block", 0.24)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = 0.065  # Stała bliska odległość
    y = -L_HUM * 0.18 * (1 - t)
    z = C_D + 0.05 + t * (L_HUM + L_FOR) * 0.74
    return np.array([x, y, z])

def push_ups_kinematics(t, levers, phase="concentric"):
    """Pompki klasyczne: dłonie oparte stabilnie na ziemi, ruch góra-dół tułowia."""
    C_W = levers.get("biacromial_width", 0.41)
    x = (C_W * 1.3) / 2
    y = 0.0
    z = 0.05 + t * 0.30
    return np.array([x, y, z])

def dips_kinematics(t, levers, phase="concentric"):
    C_W = levers.get("biacromial_width", 0.41)
    grip_x = (C_W * 1.35) / 2.0
    return np.array([grip_x, 0.0, 0.35])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_flat_barbell_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    chest_block = levers.get("chest_block", 0.240)

    weight_total = _get_weight(prof, "barbell_bench", 95.0)
    weight_per_arm = weight_total / 2.0

    m_arms_x = l_humerus * np.cos(np.radians(45)) * (1 - 0.7 * t_vals)
    m_arms_y = l_humerus * 0.30 * (1 - t_vals)**4
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_y**2)

    elbow_drop = chest_block - (l_humerus * 0.8)
    penalty = 1.0 if elbow_drop > 0 else max(0.5, 1.0 - abs(elbow_drop) * 3.0)

    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.0, pec_torque_share=0.70, mid_activation=0.85, penalty=penalty)
    act_mid = min(0.85, raw_score / 350.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"barbell", "bench", "squat_rack"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [45.0, 70.0],
            "bar_path_angle": [85.0, 95.0]
        },
        "trajectory_func": barbell_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.20, "delt_front": 0.65, "tricep_lateral": 0.75},
        "fibers": {"chest_mid": 0.85, "chest_upper": 0.25, "delt_front": 0.85, "tricep_lateral": 0.80}
    }

def evaluate_flat_dumbbell_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight = _get_weight(prof, "dumbbell_press", 32.5)

    m_arms_x = l_humerus * np.cos(np.radians(60)) * (1 - 0.85 * t_vals)
    m_arms_y = l_humerus * 0.15 * (1 - t_vals)**3
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_y**2)

    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.2, pec_torque_share=0.80, mid_activation=0.95)
    act_mid = min(0.95, raw_score / 350.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"dumbbells", "bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [45.0, 75.0],
            "elbow_flexion_bottom": [70.0, 90.0]
        },
        "trajectory_func": dumbbell_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.20, "delt_front": 0.60, "tricep_lateral": 0.50},
        "fibers": {"chest_mid": 0.95, "chest_upper": 0.30, "delt_front": 0.80, "tricep_lateral": 0.60}
    }

def evaluate_machine_chest_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "machine_hammer", 80.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, l_humerus * 0.85) * (1 - 0.4 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=True, rom_bonus=1.8, pec_torque_share=0.85, mid_activation=0.95)
    act_mid = min(0.95, raw_score / 350.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"machine_chest_press"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [55.0, 80.0],
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": machine_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.15, "delt_front": 0.50, "tricep_lateral": 0.60},
        "fibers": {"chest_mid": 0.95, "chest_upper": 0.25, "delt_front": 0.70, "tricep_lateral": 0.70}
    }

def evaluate_flat_smith_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "smith_bench", 85.0)
    weight_per_arm = weight_total / 2.0

    m_arms_x = l_humerus * np.cos(np.radians(50)) * (1 - 0.6 * t_vals)
    total_m_arms = m_arms_x
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.2, pec_torque_share=0.75, mid_activation=0.90)
    act_mid = min(0.90, raw_score / 350.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"smith_machine", "bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [50.0, 75.0],
            "vertical_bar_path_deviation": [0.0, 5.0]
        },
        "trajectory_func": smith_press_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.18, "delt_front": 0.60, "tricep_lateral": 0.70},
        "fibers": {"chest_mid": 0.90, "chest_upper": 0.25, "delt_front": 0.75, "tricep_lateral": 0.75}
    }

def evaluate_mid_cable_crossover(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight = _get_weight(prof, "cable_flyes", 27.5)

    total_m_arms = l_humerus * (1 - 0.25 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.0, pec_torque_share=0.95, mid_activation=1.00)
    act_mid = min(0.95, raw_score / 250.0)

    return {
        "cat": "Push",
        "subcat": "isolation",
        "equipment": {"cable_machine"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [70.0, 90.0],
            "elbow_flexion_bottom": [130.0, 160.0]
        },
        "trajectory_func": cable_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.15, "delt_front": 0.20},
        "fibers": {"chest_mid": 0.90, "chest_upper": 0.20, "delt_front": 0.30}
    }

def evaluate_butterfly_pec_deck(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "butterfly_weight", 65.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, l_humerus * 1.1)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=True, rom_bonus=2.4, pec_torque_share=0.98, mid_activation=1.00, is_fly=True)
    act_mid = min(1.00, raw_score / 260.0)

    return {
        "cat": "Push",
        "subcat": "isolation",
        "equipment": {"pec_deck_machine"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [75.0, 90.0],
            "elbow_stability": [85.0, 95.0]
        },
        "trajectory_func": butterfly_pec_deck_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.10, "delt_front": 0.15},
        "fibers": {"chest_mid": 1.00, "chest_upper": 0.15, "delt_front": 0.25}
    }

def evaluate_flat_dumbbell_flyes(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    weight = _get_weight(prof, "dumbbell_flyes", 18.0)

    total_m_arms = (l_humerus + l_forearm * 0.7) * (1 - t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight, has_adduction=True, rom_bonus=2.5, pec_torque_share=0.95, mid_activation=0.95, is_fly=True)
    act_mid = min(0.95, raw_score / 280.0)

    return {
        "cat": "Push",
        "subcat": "isolation",
        "equipment": {"dumbbells", "bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [65.0, 85.0],
            "elbow_flexion_constant": [130.0, 155.0]
        },
        "trajectory_func": dumbbell_flyes_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.15, "delt_front": 0.30},
        "fibers": {"chest_mid": 0.95, "chest_upper": 0.20, "delt_front": 0.40}
    }

def evaluate_hex_squeeze_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = _get_weight(prof, "hex_press", 40.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * 0.40 * (1 - 0.5 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=True, rom_bonus=0.8, pec_torque_share=0.85, mid_activation=0.95, is_hex=True)
    act_mid = min(0.92, raw_score / 280.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"dumbbells", "bench"},
        "biomechanical_bounds": {
            "dumbbell_adduction_distance": [0.0, 0.05],
            "elbow_tuck_angle": [30.0, 50.0]
        },
        "trajectory_func": hex_press_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.25, "delt_front": 0.40, "tricep_lateral": 0.70},
        "fibers": {"chest_mid": 0.90, "chest_upper": 0.30, "delt_front": 0.55, "tricep_lateral": 0.75}
    }

def evaluate_push_ups_standard(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    weight_per_arm = (body_weight * 0.65) / 2.0

    total_m_arms = l_humerus * np.cos(np.radians(45)) * (1 - 0.6 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.2, pec_torque_share=0.75, mid_activation=0.85)
    act_mid = min(0.88, raw_score / 350.0)

    return {
        "cat": "Push",
        "subcat": "bodyweight",
        "equipment": {"bodyweight", "handles"},
        "biomechanical_bounds": {
            "shoulder_abduction_angle": [45.0, 70.0],
            "torso_straight_angle": [170.0, 180.0]
        },
        "trajectory_func": push_ups_kinematics,
        "act": {"chest_mid": act_mid, "chest_upper": 0.15, "delt_front": 0.60, "tricep_lateral": 0.65},
        "fibers": {"chest_mid": 0.85, "chest_upper": 0.20, "delt_front": 0.75, "tricep_lateral": 0.70}
    }

def evaluate_chest_dips(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    weight_total = prof.get("weight_kg", 87.0) + _get_weight(prof, "dips_added", 0.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * np.cos(np.radians(30)) * (1 - 0.5 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, has_adduction=False, rom_bonus=1.5, pec_torque_share=0.75, mid_activation=0.80)

    act_mid = min(0.90, raw_score / 350.0)
    act_lower = min(0.95, raw_score / 300.0)

    return {
        "cat": "Push",
        "subcat": "compound",
        "equipment": {"dip_station"},
        "biomechanical_bounds": {
            "torso_forward_lean": [30.0, 45.0],
            "shoulder_extension_max": [0.0, 10.0]
        },
        "trajectory_func": dips_kinematics,
        "act": {"chest_lower": act_lower, "chest_mid": act_mid, "delt_front": 0.50, "tricep_lateral": 0.85},
        "fibers": {"chest_lower": 0.95, "chest_mid": 0.80, "delt_front": 0.60, "tricep_lateral": 0.90}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA ŚRODEK KLATKI
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Flat_Barbell_Press": evaluate_flat_barbell_press(user_profile),
        "Flat_Dumbbell_Press": evaluate_flat_dumbbell_press(user_profile),
        "Machine_Chest_Press": evaluate_machine_chest_press(user_profile),
        "Flat_Smith_Press": evaluate_flat_smith_press(user_profile),
        "Mid_Cable_Crossover": evaluate_mid_cable_crossover(user_profile),
        "Butterfly_Pec_Deck": evaluate_butterfly_pec_deck(user_profile),
        "Flat_Dumbbell_Flyes": evaluate_flat_dumbbell_flyes(user_profile),
        "Hex_Squeeze_Press": evaluate_hex_squeeze_press(user_profile),
        "Push_Ups_Standard": evaluate_push_ups_standard(user_profile),
        "Chest_Dips": evaluate_chest_dips(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń środka klatki. Wszystkie metryki (fibers, bounds, IK) kompletne.")
    except ImportError:
        pass