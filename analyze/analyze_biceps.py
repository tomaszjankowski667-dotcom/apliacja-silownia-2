import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_per_arm_kg, rom_bonus,
                           bicep_torque_share, bicep_activation, penalty=1.0, 
                           is_preacher=False, is_incline=False, is_cable=False, is_drag=False):
    """
    Krzywa oporu dla zginaczy ramienia (biceps/brachialis).
    Fizyka zmienia się drastycznie w zależności od wektora grawitacji i ułożenia łokcia.
    """
    tau = (weight_per_arm_kg * G * total_moment_arm) * bicep_torque_share
    
    if is_preacher:
        # Modlitewnik: najciężej na samym dole (lub lekko wyżej), na górze napięcie spada do ZERA
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-10 * t_vals))
        leverage_factor = np.cos(t_vals * np.pi / 2.5) # Szybki spadek siły w miarę podnoszenia
    elif is_incline:
        # Hantle na skosie: ekstremalne rozciągnięcie głowy długiej (łokcie za ciałem)
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.5 * np.exp(-12 * t_vals))
        # Moment maksymalny przesuwa się lekko w dół ze względu na kąt ciała
        leverage_factor = np.sin(np.clip(t_vals + 0.15, 0, 1) * np.pi) 
    elif is_cable:
        # Wyciąg: linka zapewnia ciągłe napięcie, brak martwych punktów
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-6 * t_vals))
        leverage_factor = 0.6 + 0.4 * np.sin(t_vals * np.pi)
    elif is_drag:
        # Drag curl: łokcie idą w tył, sztanga przy ciele. Bardzo krótki ROM, szczytowe napięcie głowy długiej.
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.8 * np.exp(-8 * t_vals))
        leverage_factor = np.sin(t_vals * np.pi) * 0.85
    else:
        # Klasyczne wolne ciężary (stojąc): zero na dole, szczyt przy 90 stopniach zgięcia, zero na górze
        stretch_bonus_factor = 1.0
        leverage_factor = np.sin(t_vals * np.pi)

    curve = tau * stretch_bonus_factor * leverage_factor * bicep_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def standard_curl_kinematics(t, levers, phase="concentric", grip_width_factor=1.0):
    """Baza dla klasycznych ugięć (wąsko, szeroko, młotki, nachwyt)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    # Ramię (kość ramienna) wisi prostopadle do ziemi
    shoulder_y = 0.0
    elbow_y = -L_HUM
    elbow_z = 0.0
    
    # t=0 (ręka prosta w dół), t=1 (dłoń przy barku)
    angle = np.radians(150) * t # Ugięcie od 180 do ok. 30 st.
    x = (C_W * grip_width_factor) / 2
    y = elbow_y + L_FOR * (1 - np.cos(angle))
    z = elbow_z + L_FOR * np.sin(angle)
    return np.array([x, y, z])

def incline_curl_kinematics(t, levers, phase="concentric"):
    """Ławka skośna: Łokcie wiszą daleko za linią tułowia (hiperekstensja stawu ramiennego)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 1.1) / 2
    # Łokcie cofnięte (ujemne Z)
    elbow_y = -L_HUM * 0.85
    elbow_z = -L_HUM * 0.35 
    
    angle = np.radians(140) * t
    y = elbow_y + L_FOR * (1 - np.cos(angle))
    z = elbow_z + L_FOR * np.sin(angle)
    return np.array([x, y, z])

def drag_curl_kinematics(t, levers, phase="concentric"):
    """Drag Curl: Sztanga wędruje pionowo przy ciele, łokcie uciekają mocno do tyłu."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 1.0) / 2
    # Sztanga idzie tylko w górę (Z = 0 przy brzuchu), łokcie amortyzują to z tyłu
    y = -L_HUM - L_FOR + ((L_HUM + L_FOR) * 0.6) * t
    z = 0.05 # Lekko przed klatką/brzuchem na sztywno
    return np.array([x, y, z])

def preacher_curl_kinematics(t, levers, phase="concentric"):
    """Modlitewnik: Łokcie podparte i wypchnięte mocno w przód (zgięcie barku ok. 45 st)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.9) / 2
    # Podparcie łokcia wypycha go w przód i w górę
    elbow_y = -L_HUM * 0.70
    elbow_z = L_HUM * 0.70
    
    # Skrócony zakres ruchu w modlitewniku (od pełnego wyprostu do pionu)
    angle_start = -np.radians(45) # Dłoń nisko w dół i w przód po skosie pultpitu
    angle_end = np.radians(90)    # Dłoń w pionie nad łokciem
    current_angle = angle_start + (angle_end - angle_start) * t
    
    y = elbow_y + L_FOR * np.sin(current_angle)
    z = elbow_z - L_FOR * np.cos(current_angle)
    return np.array([x, y, z])

def high_cable_curl_kinematics(t, levers, phase="concentric"):
    """High Cable (Poza front double biceps): Łokcie na wysokości barków (90 st odwodzenia)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    # Łokieć daleko na osi X
    elbow_x = (C_W / 2) + L_HUM
    elbow_y = 0.0 # Na wysokości barku
    elbow_z = 0.0
    
    # Ruch na osi X w stronę głowy
    angle = np.radians(135) * t
    x = elbow_x - L_FOR * (1 - np.cos(angle))
    y = elbow_y + L_FOR * 0.2 * np.sin(angle) # Zwykle łokieć jest lekko wyżej niż dłoń
    z = 0.0
    return np.array([x, y, z])

def concentration_curl_kinematics(t, levers, phase="concentric"):
    """Uginanie skoncentrowane: Tułów pochylony, ramię wisi pionowo, podparte o udo."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.5) / 2 # Dłoń idzie w kierunku klatki piersiowej na środek
    elbow_y = -L_HUM * 0.95
    elbow_z = 0.10
    
    angle = np.radians(140) * t
    y = elbow_y + L_FOR * (1 - np.cos(angle))
    z = elbow_z + L_FOR * np.sin(angle)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_incline_dumbbell_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "incline_db_curl", 15.0)
    total_m_arms = l_forearm * np.sin(np.clip(t_vals + 0.15, 0, 1) * np.pi)

    # Król rozciągnięcia głowy długiej
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=2.5, bicep_torque_share=0.95, bicep_activation=0.98, is_incline=True)
    act_bicep = min(1.00, raw_score / 160.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"dumbbells", "incline_bench"},
        "biomechanical_bounds": {
            "shoulder_extension_start": [15.0, 30.0], # Łokcie z tyłu
            "elbow_flexion_end": [130.0, 150.0]
        },
        "trajectory_func": incline_curl_kinematics,
        "act": {"biceps": act_bicep},
        "fibers": {"biceps_long_head": 1.00, "biceps_short_head": 0.60, "brachialis": 0.50}
    }

def evaluate_cable_behind_back_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "cable_behind_back", 15.0)
    total_m_arms = l_forearm * (0.6 + 0.4 * np.sin(t_vals * np.pi))

    # Podobnie jak ławka skośna, ale wyciąg daje opór na całym zakresie (stałe napięcie wstępne)
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=2.0, bicep_torque_share=1.00, bicep_activation=1.00, is_cable=True)
    act_bicep = min(1.00, raw_score / 170.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"cable_machine", "single_handle"},
        "biomechanical_bounds": {
            "shoulder_extension_start": [10.0, 25.0],
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": incline_curl_kinematics, # Kinematyka ta sama co Incline (łokcie z tyłu)
        "act": {"biceps": act_bicep},
        "fibers": {"biceps_long_head": 1.00, "biceps_short_head": 0.70, "brachialis": 0.40}
    }

def evaluate_close_grip_barbell_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "close_grip_curl", 30.0)
    weight_per_arm = weight / 2.0
    total_m_arms = l_forearm * np.sin(t_vals * np.pi)

    # Wąski chwyt na sztandze rotuje stawy ramienne i mocniej obciąża głowę długą
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.8, bicep_torque_share=0.85, bicep_activation=0.90)
    act_bicep = min(0.92, raw_score / 180.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"barbell", "ez_bar"},
        "biomechanical_bounds": {
            "grip_width_ratio": [0.6, 0.9],
            "elbow_flexion_end": [130.0, 150.0]
        },
        "trajectory_func": lambda t, lev, p="concentric": standard_curl_kinematics(t, lev, p, grip_width_factor=0.6),
        "act": {"biceps": act_bicep, "forearms": 0.40},
        "fibers": {"biceps_long_head": 0.95, "biceps_short_head": 0.60, "brachioradialis": 0.50}
    }

def evaluate_wide_grip_barbell_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "wide_grip_curl", 30.0)
    weight_per_arm = weight / 2.0
    total_m_arms = l_forearm * np.sin(t_vals * np.pi)

    # Szeroki chwyt zdejmuje pracę z głowy długiej, pompując całą krew do głowy krótkiej (wewnętrznej)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.8, bicep_torque_share=0.85, bicep_activation=0.90)
    act_bicep = min(0.92, raw_score / 180.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"barbell"},
        "biomechanical_bounds": {
            "grip_width_ratio": [1.3, 1.8],
            "elbow_flexion_end": [130.0, 150.0]
        },
        "trajectory_func": lambda t, lev, p="concentric": standard_curl_kinematics(t, lev, p, grip_width_factor=1.5),
        "act": {"biceps": act_bicep, "forearms": 0.35},
        "fibers": {"biceps_short_head": 1.00, "biceps_long_head": 0.40, "brachioradialis": 0.40}
    }

def evaluate_drag_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "drag_curl", 35.0)
    weight_per_arm = weight / 2.0
    
    total_m_arms = l_forearm * np.sin(t_vals * np.pi) * 0.85
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.5, bicep_torque_share=0.95, bicep_activation=0.90, is_drag=True)
    act_bicep = min(0.92, raw_score / 170.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"barbell", "smith_machine"},
        "biomechanical_bounds": {
            "shoulder_extension_dynamic": [0.0, 35.0], # Ruchome łokcie w tył to sedno Drag Curla
            "bar_path_verticality": [95.0, 100.0] # Prosta linia tuż przy ciele
        },
        "trajectory_func": drag_curl_kinematics,
        "act": {"biceps": act_bicep, "delt_rear": 0.40},
        "fibers": {"biceps_long_head": 1.00, "biceps_short_head": 0.50, "brachialis": 0.40}
    }

def evaluate_preacher_curl_ez(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "preacher_ez", 30.0)
    weight_per_arm = weight / 2.0
    total_m_arms = l_forearm * np.cos(t_vals * np.pi / 2.5)

    # Modlitewnik potwornie obciąża przyczepy przy pełnym rozciągnięciu (wysokie siły tnące), ale gasi napięcie w pionie
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=2.0, bicep_torque_share=1.00, bicep_activation=0.95, is_preacher=True)
    act_bicep = min(0.98, raw_score / 180.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"preacher_bench", "ez_bar"},
        "biomechanical_bounds": {
            "shoulder_flexion_constant": [30.0, 55.0], # Wypchnięte barki w przód skracają głowę długą
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": preacher_curl_kinematics,
        "act": {"biceps": act_bicep, "forearms": 0.50},
        "fibers": {"biceps_short_head": 1.00, "brachialis": 0.90, "biceps_long_head": 0.20} # Głowa długa niemal wyłączona z racji zgięcia barku
    }

def evaluate_preacher_curl_machine(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "preacher_machine", 40.0)
    weight_per_arm = weight / 2.0
    total_m_arms = np.full_like(t_vals, l_forearm * 0.8) # Krzywka maszyny naprawia błąd wolnego ciężaru (daje napięcie na górze)

    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.5, bicep_torque_share=1.00, bicep_activation=1.00, is_cable=True) # Traktujemy jak kabel
    act_bicep = min(1.00, raw_score / 180.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"preacher_machine"},
        "biomechanical_bounds": {
            "shoulder_flexion_constant": [30.0, 55.0],
            "elbow_flexion_end": [120.0, 140.0]
        },
        "trajectory_func": preacher_curl_kinematics,
        "act": {"biceps": act_bicep},
        "fibers": {"biceps_short_head": 1.00, "brachialis": 0.95, "biceps_long_head": 0.20}
    }

def evaluate_high_cable_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "high_cable_curl", 15.0)
    total_m_arms = l_forearm * (0.6 + 0.4 * np.sin(t_vals * np.pi))

    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.8, bicep_torque_share=0.95, bicep_activation=1.00, is_cable=True)
    act_bicep = min(0.95, raw_score / 160.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"cable_machine", "single_handle"},
        "biomechanical_bounds": {
            "shoulder_abduction_constant": [85.0, 95.0], # Ręce wysoko w bok
            "elbow_stability": [90.0, 100.0]
        },
        "trajectory_func": high_cable_curl_kinematics,
        "act": {"biceps": act_bicep},
        "fibers": {"biceps_short_head": 1.00, "biceps_long_head": 0.50, "brachialis": 0.40}
    }

def evaluate_concentration_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "concentration_curl", 15.0)
    total_m_arms = l_forearm * np.sin(t_vals * np.pi)

    # 100% izolacji (badania EMG pokazują tu często najwyższą aktywność, bo nie ma jak oszukać ciałem)
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.0, bicep_torque_share=1.00, bicep_activation=1.00)
    act_bicep = min(0.98, raw_score / 150.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"dumbbell"},
        "biomechanical_bounds": {
            "elbow_stability": [98.0, 100.0], # Zablokowane o wewn. stronę uda
            "torso_forward_lean": [70.0, 90.0]
        },
        "trajectory_func": concentration_curl_kinematics,
        "act": {"biceps": act_bicep},
        "fibers": {"biceps_short_head": 0.95, "brachialis": 0.85, "biceps_long_head": 0.40}
    }

def evaluate_hammer_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "hammer_curl", 20.0)
    total_m_arms = l_forearm * np.sin(t_vals * np.pi)

    # Chwyt młotkowy (neutralny) mechanicznie wygasza supinację bicepsa. 
    # Rolę przejmuje potężny mięsień ramienno-promieniowy i ramienny, przez co można podnieść więcej kg.
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.5, bicep_torque_share=0.85, bicep_activation=0.90)
    act_arm = min(0.95, raw_score / 180.0)

    return {
        "cat": "Arms",
        "subcat": "compound_isolation", # Angażuje całe ramię i przedramię
        "equipment": {"dumbbells"},
        "biomechanical_bounds": {
            "forearm_pronation": [45.0, 60.0], # Chwyt neutralny
            "elbow_flexion_end": [120.0, 140.0]
        },
        "trajectory_func": standard_curl_kinematics,
        "act": {"forearms": act_arm, "biceps": 0.60},
        "fibers": {"brachioradialis": 1.00, "brachialis": 1.00, "biceps_long_head": 0.50, "biceps_short_head": 0.20}
    }

def evaluate_cable_hammer_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "cable_hammer_curl", 25.0)
    total_m_arms = l_forearm * (0.6 + 0.4 * np.sin(t_vals * np.pi))

    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.0, bicep_torque_share=0.90, bicep_activation=0.95, is_cable=True)
    act_arm = min(0.98, raw_score / 170.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"cable_machine", "rope"},
        "biomechanical_bounds": {
            "forearm_pronation": [45.0, 60.0],
            "torso_stability": [90.0, 100.0]
        },
        "trajectory_func": standard_curl_kinematics,
        "act": {"forearms": act_arm, "biceps": 0.65},
        "fibers": {"brachioradialis": 1.00, "brachialis": 1.00, "biceps_long_head": 0.60}
    }

def evaluate_reverse_curl(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "reverse_curl", 25.0)
    weight_per_arm = weight / 2.0
    total_m_arms = l_forearm * np.sin(t_vals * np.pi)

    # Całkowita pronacja (nachwyt) w 90% wyłącza biceps z pracy.
    # To jest ćwiczenie stricte na mięsień ramienno-promieniowy (przedramię) i ramienny.
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.0, bicep_torque_share=0.70, bicep_activation=0.85)
    act_arm = min(0.92, raw_score / 150.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"barbell", "ez_bar"},
        "biomechanical_bounds": {
            "forearm_pronation": [80.0, 100.0], # Pełny nachwyt
            "elbow_flexion_end": [110.0, 130.0] # Ruch zatrzymuje się wcześniej fizjologicznie
        },
        "trajectory_func": standard_curl_kinematics,
        "act": {"forearms": act_arm, "biceps": 0.30},
        "fibers": {"brachioradialis": 1.00, "brachialis": 0.90, "extensor_carpi": 0.70, "biceps_short_head": 0.10}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA ZGINACZE RAMIENIA
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Incline_Dumbbell_Curl": evaluate_incline_dumbbell_curl(user_profile),
        "Cable_Behind_Back_Curl": evaluate_cable_behind_back_curl(user_profile),
        "Close_Grip_Barbell_Curl": evaluate_close_grip_barbell_curl(user_profile),
        "Wide_Grip_Barbell_Curl": evaluate_wide_grip_barbell_curl(user_profile),
        "Drag_Curl": evaluate_drag_curl(user_profile),
        "Preacher_Curl_EZ_Bar": evaluate_preacher_curl_ez(user_profile),
        "Preacher_Curl_Machine": evaluate_preacher_curl_machine(user_profile),
        "High_Cable_Curl": evaluate_high_cable_curl(user_profile),
        "Concentration_Curl": evaluate_concentration_curl(user_profile),
        "Hammer_Curl": evaluate_hammer_curl(user_profile),
        "Cable_Hammer_Curl": evaluate_cable_hammer_curl(user_profile),
        "Reverse_Curl": evaluate_reverse_curl(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na zginacze ramienia (bicepsy/brachialis). Wszystkie metryki kompletne.")
    except ImportError:
        pass