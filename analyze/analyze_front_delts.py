import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           delt_torque_share, delt_activation, penalty=1.0, 
                           is_press=False, is_front_raise_fw=False, is_cable=False):
    """
    Krzywa oporu dla przedniego aktonu barku. 
    Diametralnie inna fizyka dla wyciskań, wznosów wolnym ciężarem i wyciągów.
    """
    tau = (weight_kg * G * total_moment_arm) * delt_torque_share
    
    if is_press:
        # Wyciskania nad głowę: najciężej na dole, stretch bonus duży
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.4 * t_vals # Ciężej na dole, na górze zablokowane stawy przejmują część sił
    elif is_front_raise_fw:
        # Wznosy wolnym ciężarem (hantle/sztanga): na dole ramię siły = 0, napięcie zerowe. 
        # Największe uderzenie w peak contraction (kąt 90 stopni do tułowia)
        stretch_bonus_factor = 1.0 
        leverage_factor = 0.1 + 0.9 * np.sin(t_vals * np.pi / 2) 
    elif is_cable:
        # Wznosy na wyciągu dolnym: lina ciągnie pod kątem, dając napięcie od samego dołu
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.8 * np.exp(-6 * t_vals))
        leverage_factor = 0.6 + 0.4 * np.sin(t_vals * np.pi / 2)
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * delt_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def ohp_kinematics(t, levers, phase="concentric"):
    """Wyciskanie żołnierskie (OHP): Tor sztangi pionowy, sztanga omija twarz."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 1.4) / 2
    # Sztanga z obojczyków (t=0) nad głowę (t=1)
    y = L_HUM * 0.20 + (L_HUM + L_FOR) * 0.85 * t
    # Lekki łuk: na dole przed twarzą, na górze nad środkiem ciężkości ciała (w linii uszu)
    z = 0.15 * (1 - t) 
    return np.array([x, y, z])

def dumbbell_shoulder_press_kinematics(t, levers, phase="concentric"):
    """Wyciskanie hantli: konwergujący (zbiegający się) łuk nad głową."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    # Hantle startują szerzej (t=0), kończą wąsko nad głową (t=1)
    x = (C_W / 2 + L_HUM * 0.6) * (1 - t) + 0.10 * t
    y = L_HUM * 0.25 + (L_HUM + L_FOR) * 0.85 * t
    z = 0.10 * (1 - t)
    return np.array([x, y, z])

def arnold_press_kinematics(t, levers, phase="concentric"):
    """Wyciskanie Arnolda: ogromna rotacja na dole, łokcie przed klatką."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    # Hantle przed twarzą złączone (t=0), otwierają się i wędrują nad głowę (t=1)
    x = 0.15 * (1 - t) + 0.10 * t if t > 0.4 else 0.10 + 0.25 * t # Symulacja łuku otwierającego
    y = L_HUM * 0.10 + (L_HUM + L_FOR) * 0.90 * t
    # Mocno wysunięte do przodu na starcie
    z = 0.25 * (1 - t)
    return np.array([x, y, z])

def smith_shoulder_press_kinematics(t, levers, phase="concentric"):
    """Wyciskanie na maszynie Smitha siedząc: sztywna, często pionowa prowadnica."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 1.5) / 2
    y = L_HUM * 0.20 + (L_HUM + L_FOR) * 0.85 * t
    # Sztywna prowadnica (często lekki skos lub pion), sztanga nie wymija twarzy tak łatwo
    z = 0.12 - 0.05 * t 
    return np.array([x, y, z])

def machine_shoulder_press_kinematics(t, levers, phase="concentric"):
    """Wyciskanie na maszynie Hammer: krzywka maszyny narzuca optymalny zbiegający tor."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = ((C_W / 2) + L_HUM * 0.7) * (1 - t) + 0.15 * t
    y = L_HUM * 0.15 + (L_HUM + L_FOR) * 0.80 * t
    z = 0.10 * (1 - t)
    return np.array([x, y, z])

def dumbbell_front_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy ramion w przód z hantlami: ramię proste, czyste zgięcie stawu ramiennego."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    x = (C_W * 1.1) / 2
    # Ręka zwisa w dół (-90 st względem poziomu) -> podnosi się do poziomu (0 st)
    angle = np.radians(90) * (1 - t)
    y = -r * np.cos(angle)
    z = r * np.sin(angle)
    return np.array([x, y, z])

def barbell_front_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy w przód sztangą/talerzem: złączone oburącz z przodu."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.95 # Lekko ugięte łokcie przy sztandze/talerzu
    
    x = (C_W * 0.8) / 2 # Wąski chwyt z przodu
    angle = np.radians(85) * (1 - t)
    y = -r * np.cos(angle)
    z = r * np.sin(angle)
    return np.array([x, y, z])

def cable_front_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy w przód na wyciągu: tożsama kinematyka z wolnym ciężarem, inna fizyka napięcia."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    x = (C_W * 1.0) / 2
    angle = np.radians(80) * (1 - t) + np.radians(10) * t # Kabel narzuca lekkie napięcie wstępne z przodu
    y = -r * np.cos(angle)
    z = r * np.sin(angle)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_ohp(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    barbell_weight = _get_weight(prof, "ohp_weight", 60.0)
    weight_per_arm = barbell_weight / 2.0

    # Ramię momentu na poziomie stawu barkowego w osi strzałkowej/czołowej
    m_arms_x = l_humerus * np.cos(np.radians(60)) * (1 - 0.7 * t_vals)
    m_arms_z = l_humerus * 0.30 * (1 - t_vals)
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_z**2)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.2, delt_torque_share=0.85, delt_activation=0.95, is_press=True)
    act_front = min(0.95, raw_score / 250.0)

    return {
        "cat": "Shoulders",
        "subcat": "compound",
        "equipment": {"barbell", "squat_rack"},
        "biomechanical_bounds": {
            "shoulder_flexion_top": [160.0, 180.0],
            "torso_lean_angle": [0.0, 20.0], # Lekkie odchylenie dolnego odcinka
            "bar_path_angle": [85.0, 95.0]
        },
        "trajectory_func": ohp_kinematics,
        "act": {"delt_front": act_front, "delt_lateral": 0.40, "tricep_lateral": 0.85, "chest_upper": 0.40},
        "fibers": {"delt_front": 0.95, "delt_lateral": 0.45, "tricep_lateral": 0.90, "chest_upper": 0.50}
    }

def evaluate_dumbbell_shoulder_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    dumbbell_weight = _get_weight(prof, "db_shoulder_press", 25.0)

    m_arms_x = l_humerus * np.cos(np.radians(70)) * (1 - 0.8 * t_vals)
    m_arms_z = l_humerus * 0.20 * (1 - t_vals)
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_z**2)
    
    raw_score = calc_raw_physics_score(total_m_arms, dumbbell_weight, rom_bonus=1.4, delt_torque_share=0.90, delt_activation=1.00, is_press=True)
    act_front = min(1.00, raw_score / 230.0)

    return {
        "cat": "Shoulders",
        "subcat": "compound",
        "equipment": {"dumbbells", "bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_bottom": [60.0, 90.0],
            "elbow_flexion_bottom": [75.0, 95.0]
        },
        "trajectory_func": dumbbell_shoulder_press_kinematics,
        "act": {"delt_front": act_front, "delt_lateral": 0.50, "tricep_lateral": 0.70},
        "fibers": {"delt_front": 1.00, "delt_lateral": 0.55, "tricep_lateral": 0.75}
    }

def evaluate_arnold_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    dumbbell_weight = _get_weight(prof, "arnold_press", 20.0)

    # Ogromne ramię momentu do przodu na samym dole z powodu specyficznej pozycji
    m_arms_x = l_humerus * np.cos(np.radians(90)) * (1 - 0.5 * t_vals)
    m_arms_z = l_humerus * 0.60 * (1 - t_vals)
    total_m_arms = np.sqrt(m_arms_x**2 + m_arms_z**2)
    
    raw_score = calc_raw_physics_score(total_m_arms, dumbbell_weight, rom_bonus=1.8, delt_torque_share=0.95, delt_activation=1.00, is_press=True)
    act_front = min(1.00, raw_score / 220.0)

    return {
        "cat": "Shoulders",
        "subcat": "compound",
        "equipment": {"dumbbells", "bench"},
        "biomechanical_bounds": {
            "shoulder_rotation_dynamic": [90.0, 180.0], # Supinacja do pronacji
            "shoulder_flexion_start": [70.0, 90.0]
        },
        "trajectory_func": arnold_press_kinematics,
        "act": {"delt_front": act_front, "delt_lateral": 0.60, "chest_upper": 0.30},
        "fibers": {"delt_front": 1.00, "delt_lateral": 0.65, "chest_upper": 0.40}
    }

def evaluate_smith_shoulder_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    smith_weight = _get_weight(prof, "smith_shoulder_press", 55.0)
    weight_per_arm = smith_weight / 2.0

    total_m_arms = l_humerus * np.cos(np.radians(65)) * (1 - 0.6 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.0, delt_torque_share=0.85, delt_activation=0.90, is_press=True)
    act_front = min(0.92, raw_score / 240.0)

    return {
        "cat": "Shoulders",
        "subcat": "compound",
        "equipment": {"smith_machine", "bench"},
        "biomechanical_bounds": {
            "shoulder_flexion_top": [160.0, 180.0],
            "vertical_bar_path_deviation": [0.0, 2.0]
        },
        "trajectory_func": smith_shoulder_press_kinematics,
        "act": {"delt_front": act_front, "delt_lateral": 0.35, "tricep_lateral": 0.85},
        "fibers": {"delt_front": 0.90, "delt_lateral": 0.40, "tricep_lateral": 0.90}
    }

def evaluate_machine_shoulder_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    machine_weight = _get_weight(prof, "machine_shoulder_press", 65.0)
    weight_per_arm = machine_weight / 2.0

    total_m_arms = np.full_like(t_vals, l_humerus * 0.45 * (1 - 0.4 * t_vals))
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.1, delt_torque_share=0.90, delt_activation=0.95, is_press=True)
    act_front = min(0.95, raw_score / 250.0)

    return {
        "cat": "Shoulders",
        "subcat": "compound",
        "equipment": {"shoulder_press_machine"},
        "biomechanical_bounds": {
            "shoulder_flexion_top": [150.0, 175.0],
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": machine_shoulder_press_kinematics,
        "act": {"delt_front": act_front, "delt_lateral": 0.45, "tricep_lateral": 0.70},
        "fibers": {"delt_front": 0.95, "delt_lateral": 0.50, "tricep_lateral": 0.75}
    }

def evaluate_dumbbell_front_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    dumbbell_weight = _get_weight(prof, "db_front_raise", 12.5)

    # Ramię siły równe długości całej ręki w poziomie
    total_m_arms = (l_humerus + l_forearm) * np.sin(t_vals * np.pi / 2)
    
    raw_score = calc_raw_physics_score(total_m_arms, dumbbell_weight, rom_bonus=0.0, delt_torque_share=1.00, delt_activation=0.90, is_front_raise_fw=True)
    act_front = min(0.90, raw_score / 150.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"dumbbells"},
        "biomechanical_bounds": {
            "shoulder_flexion_end": [85.0, 105.0], # Ruch kończy się na wysokości oczu/barków
            "elbow_flexion_constant": [160.0, 180.0]
        },
        "trajectory_func": dumbbell_front_raise_kinematics,
        "act": {"delt_front": act_front},
        "fibers": {"delt_front": 1.00, "chest_upper": 0.20} # Czysta izolacja
    }

def evaluate_barbell_front_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    barbell_weight = _get_weight(prof, "bb_front_raise", 25.0)
    weight_per_arm = barbell_weight / 2.0

    total_m_arms = (l_humerus + l_forearm) * 0.95 * np.sin(t_vals * np.pi / 2)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.0, delt_torque_share=1.00, delt_activation=0.92, is_front_raise_fw=True)
    act_front = min(0.92, raw_score / 150.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"barbell", "weight_plate"},
        "biomechanical_bounds": {
            "shoulder_flexion_end": [85.0, 105.0],
            "torso_stability": [90.0, 100.0] # Brak zarzucania plecami
        },
        "trajectory_func": barbell_front_raise_kinematics,
        "act": {"delt_front": act_front},
        "fibers": {"delt_front": 1.00, "chest_upper": 0.25}
    }

def evaluate_cable_front_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    cable_weight = _get_weight(prof, "cable_front_raise", 15.0)

    # Wyciąg zapewnia napięcie nawet gdy ręka wisi, dając lepszy profil oporu
    total_m_arms = (l_humerus + l_forearm) * (0.4 + 0.6 * np.sin(t_vals * np.pi / 2))
    
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=1.0, delt_torque_share=1.00, delt_activation=0.98, is_cable=True)
    act_front = min(0.95, raw_score / 180.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"cable_machine", "straight_bar_or_rope"},
        "biomechanical_bounds": {
            "shoulder_flexion_end": [85.0, 105.0],
            "elbow_flexion_constant": [165.0, 180.0]
        },
        "trajectory_func": cable_front_raise_kinematics,
        "act": {"delt_front": act_front},
        "fibers": {"delt_front": 1.00, "chest_upper": 0.15}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA PRZÓD BARKU
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Overhead_Press_Barbell": evaluate_ohp(user_profile),
        "Dumbbell_Shoulder_Press": evaluate_dumbbell_shoulder_press(user_profile),
        "Arnold_Press": evaluate_arnold_press(user_profile),
        "Smith_Shoulder_Press": evaluate_smith_shoulder_press(user_profile),
        "Machine_Shoulder_Press": evaluate_machine_shoulder_press(user_profile),
        "Dumbbell_Front_Raise": evaluate_dumbbell_front_raise(user_profile),
        "Barbell_Front_Raise": evaluate_barbell_front_raise(user_profile),
        "Cable_Front_Raise": evaluate_cable_front_raise(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na przedni akton barków. Wszystkie metryki kompletne.")
    except ImportError:
        pass