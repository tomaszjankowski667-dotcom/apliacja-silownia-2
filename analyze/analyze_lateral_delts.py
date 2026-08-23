import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           delt_torque_share, delt_activation, penalty=1.0, 
                           is_fw_raise=False, is_cable=False, is_lying_incline=False, 
                           is_machine=False, is_upright_row=False):
    """
    Krzywa oporu dla bocznego aktonu barku.
    Główną zmienną jest profil oporu, zależny od użytego sprzętu (hantle vs wyciąg/maszyna).
    """
    tau = (weight_kg * G * total_moment_arm) * delt_torque_share
    
    if is_fw_raise:
        # Hantle stojąc: Ramię siły wynosi 0 na dole i rośnie do maksimum na górze
        stretch_bonus_factor = 1.0 
        leverage_factor = np.sin(t_vals * np.pi / 2) # Od 0 do 1
    elif is_lying_incline:
        # Leżąc bokiem na ławce skośnej: Grawitacja uderza prostopadle znacznie wcześniej
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-8 * t_vals)) # Fantastyczne rozciągnięcie pod obciążeniem
        leverage_factor = np.sin(np.clip(t_vals + 0.3, 0, 1) * np.pi / 2)
    elif is_cable:
        # Wyciąg: Zapewnia napięcie od samego dołu (linka ciągnie w bok/skos)
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-6 * t_vals))
        leverage_factor = 0.5 + 0.5 * np.sin(t_vals * np.pi / 2)
    elif is_machine:
        # Maszyna z krzywką: Zapewnia stałe napięcie w całym zakresie ruchu
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)
    elif is_upright_row:
        # Podciąganie wzdłuż tułowia: Ruch złożony, wielostawowy, cięższy na dole
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.2 * t_vals
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * delt_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def lateral_raise_fw_kinematics(t, levers, phase="concentric"):
    """Wznosy bokiem z hantlami stojąc/siedząc: Odwodzenie w płaszczyźnie czołowej/łopatkowej."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    # Kąt odwodzenia: 0 (ręce wzdłuż ciała, -90 st względem ramienia) do 90 st (poziomo)
    angle = np.radians(90) * (1 - t)
    
    x = (C_W / 2) + r * np.sin(angle)
    y = -r * np.cos(angle)
    # Lekkie wysunięcie do przodu (płaszczyzna łopatkowa) dla bezpieczeństwa stawu
    z = r * 0.15 * t 
    return np.array([x, y, z])

def cable_lateral_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy bokiem z wyciągiem dolnym: Podobny tor do hantli, ale kabel krzyżuje się przed/za ciałem."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    angle = np.radians(90) * (1 - t)
    x = (C_W / 2) + r * np.sin(angle)
    y = -r * np.cos(angle)
    z = r * 0.10 * t
    return np.array([x, y, z])

def lying_incline_lateral_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy leżąc bokiem: Tułów ułożony na skosie, ramię pracuje w odwodzeniu relatywnym do tułowia."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    # Skos ławki dodaje stałe przesunięcie kątowe (np. 30-45 stopni)
    # Względem globalnego układu Y (grawitacja), ramię odrywa się szybciej
    angle = np.radians(90) * (1 - t)
    x = (C_W / 2) + r * np.sin(angle)
    y = -r * np.cos(angle)
    z = 0.0
    return np.array([x, y, z])

def machine_lateral_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy na maszynie: Stały, wymuszony tor, zgięte łokcie (punkt podparcia na przedramieniu)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    # Maszyna ma pady opierające się na przedramionach/łokciach, skracając ramię siły
    r = L_HUM * 0.95 
    
    angle = np.radians(90) * (1 - t)
    x = (C_W / 2) + r * np.sin(angle)
    y = -r * np.cos(angle)
    z = 0.0
    return np.array([x, y, z])

def upright_row_kinematics(t, levers, phase="concentric"):
    """Podciąganie wzdłuż tułowia: Prowadzenie sztangi/kablówki pionowo, łokcie idą wysoko na zewnątrz."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 0.8) / 2 # Chwyt na szerokość barków lub lekko węziej
    # Sztanga (dłonie) idzie z dołu (na udach) w kierunku obojczyków/mostka
    y = -L_HUM * 1.5 * (1 - t)
    # Blisko ciała
    z = 0.10 
    return np.array([x, y, z])

def y_raise_kinematics(t, levers, phase="concentric"):
    """Y-Raise (wznosy w Y): Ręce unoszone w skos w górę i na zewnątrz (często leżąc przodem na skosie)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    # Ruch do kąta ok. 135 stopni względem tułowia (kształt litery Y)
    angle_y = np.radians(135) * t
    angle_x = np.radians(45) # Szeroko na zewnątrz
    
    x = (C_W / 2) + r * np.sin(angle_y) * np.sin(angle_x)
    y = -r * np.cos(angle_y)
    z = r * np.sin(angle_y) * np.cos(angle_x)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_dumbbell_lateral_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    dumbbell_weight = _get_weight(prof, "db_lateral_raise", 12.5)

    # Ramię siły w poziomie równe długości całej ręki (pomniejszone przez zgięcie w łokciu)
    effective_length = l_humerus + l_forearm * 0.9
    total_m_arms = effective_length * np.sin(t_vals * np.pi / 2)
    
    raw_score = calc_raw_physics_score(total_m_arms, dumbbell_weight, rom_bonus=0.0, delt_torque_share=0.95, delt_activation=0.90, is_fw_raise=True)
    act_lateral = min(0.95, raw_score / 150.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"dumbbells"},
        "biomechanical_bounds": {
            "shoulder_abduction_end": [80.0, 100.0], # Unoszenie do poziomu
            "elbow_flexion_constant": [150.0, 175.0] # Lekko zgięte łokcie chronią staw
        },
        "trajectory_func": lateral_raise_fw_kinematics,
        "act": {"delt_lateral": act_lateral, "traps_upper": 0.40},
        "fibers": {"delt_lateral": 1.00, "delt_front": 0.30, "traps_upper": 0.60, "supraspinatus": 0.80}
    }

def evaluate_cable_lateral_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    cable_weight = _get_weight(prof, "cable_lateral_raise", 10.0)

    effective_length = l_humerus + l_forearm
    total_m_arms = effective_length * (0.5 + 0.5 * np.sin(t_vals * np.pi / 2))
    
    # Wyciąg daje bezcenne napięcie w dolnej fazie
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=1.5, delt_torque_share=0.98, delt_activation=0.95, is_cable=True)
    act_lateral = min(1.00, raw_score / 140.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"cable_machine", "handle"},
        "biomechanical_bounds": {
            "shoulder_abduction_end": [85.0, 100.0],
            "torso_stability": [90.0, 100.0]
        },
        "trajectory_func": cable_lateral_raise_kinematics,
        "act": {"delt_lateral": act_lateral, "traps_upper": 0.30},
        "fibers": {"delt_lateral": 1.00, "delt_front": 0.25, "supraspinatus": 0.90}
    }

def evaluate_lying_incline_lateral_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    dumbbell_weight = _get_weight(prof, "lying_incline_lateral", 8.0)

    effective_length = l_humerus + l_forearm
    total_m_arms = effective_length * np.sin(np.clip(t_vals + 0.3, 0, 1) * np.pi / 2)
    
    # Genialne uderzenie w peak i stretch dzięki nachyleniu tułowia
    raw_score = calc_raw_physics_score(total_m_arms, dumbbell_weight, rom_bonus=2.0, delt_torque_share=1.00, delt_activation=0.98, is_lying_incline=True)
    act_lateral = min(1.00, raw_score / 120.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"dumbbell", "incline_bench"},
        "biomechanical_bounds": {
            "shoulder_abduction_end": [70.0, 95.0],
            "torso_stability": [95.0, 100.0] # Ławka stabilizuje ciało, zero bujania
        },
        "trajectory_func": lying_incline_lateral_raise_kinematics,
        "act": {"delt_lateral": act_lateral},
        "fibers": {"delt_lateral": 1.00, "supraspinatus": 0.95} # Najsilniejsza izolacja na zewnątrz
    }

def evaluate_machine_lateral_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    machine_weight = _get_weight(prof, "machine_lateral_raise", 35.0)
    weight_per_arm = machine_weight / 2.0

    # Dźwignia skrócona do łokci (brak przedramienia w obliczeniach momentu)
    total_m_arms = np.full_like(t_vals, l_humerus * 0.9)
    
    # Maszyna z krzywką wyrównuje opór przez całą amplitudę
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.0, delt_torque_share=0.95, delt_activation=0.95, is_machine=True)
    act_lateral = min(0.95, raw_score / 160.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"lateral_raise_machine"},
        "biomechanical_bounds": {
            "shoulder_abduction_end": [80.0, 95.0],
            "elbow_flexion_constant": [90.0, 100.0] # Kąt prosty podpartych rąk
        },
        "trajectory_func": machine_lateral_raise_kinematics,
        "act": {"delt_lateral": act_lateral, "traps_upper": 0.35},
        "fibers": {"delt_lateral": 1.00, "supraspinatus": 0.85, "traps_upper": 0.50}
    }

def evaluate_upright_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    barbell_weight = _get_weight(prof, "upright_row", 40.0)
    weight_per_arm = barbell_weight / 2.0

    total_m_arms = l_humerus * (0.8 - 0.4 * t_vals)
    
    # Złożone, wielostawowe ćwiczenie, angażujące również kaptury (czworoboczny grzbietu)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.8, delt_torque_share=0.75, delt_activation=0.90, is_upright_row=True)
    act_lateral = min(0.92, raw_score / 180.0)

    return {
        "cat": "Shoulders",
        "subcat": "compound",
        "equipment": {"barbell", "cable_machine", "dumbbells"},
        "biomechanical_bounds": {
            "shoulder_abduction_end": [75.0, 95.0], # Łokcie szeroko i wysoko
            "bar_path_angle": [85.0, 95.0]
        },
        "trajectory_func": upright_row_kinematics,
        "act": {"delt_lateral": act_lateral, "traps_upper": 0.85, "biceps": 0.50},
        "fibers": {"delt_lateral": 0.95, "traps_upper": 1.00, "delt_front": 0.40, "biceps_brachii": 0.60}
    }

def evaluate_y_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    cable_weight = _get_weight(prof, "y_raise_cable", 12.5)

    effective_length = l_humerus + l_forearm
    total_m_arms = effective_length * np.sin(t_vals * np.pi) # Peak pośrodku ruchu, spada na samej górze ze względu na pion
    
    # Y-Raise jest genialne dla równowagi strukturalnej barku (boczny + tył + dolne partie kaptura)
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=1.5, delt_torque_share=0.85, delt_activation=0.95, is_cable=True)
    act_lateral = min(0.95, raw_score / 150.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"cable_machine", "incline_bench"},
        "biomechanical_bounds": {
            "shoulder_elevation_end": [120.0, 140.0], # Wysokie prowadzenie ramion (litera Y)
            "torso_stability": [90.0, 100.0]
        },
        "trajectory_func": y_raise_kinematics,
        "act": {"delt_lateral": act_lateral, "delt_rear": 0.70, "traps_lower": 0.85},
        "fibers": {"delt_lateral": 0.90, "delt_rear": 0.80, "traps_lower": 1.00, "supraspinatus": 0.90}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA BOCZNY BARKU
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Dumbbell_Lateral_Raise": evaluate_dumbbell_lateral_raise(user_profile),
        "Cable_Lateral_Raise": evaluate_cable_lateral_raise(user_profile),
        "Lying_Incline_Lateral_Raise": evaluate_lying_incline_lateral_raise(user_profile),
        "Machine_Lateral_Raise": evaluate_machine_lateral_raise(user_profile),
        "Upright_Row": evaluate_upright_row(user_profile),
        "Y_Raise": evaluate_y_raise(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na boczny akton barków. Wszystkie metryki kompletne.")
    except ImportError:
        pass