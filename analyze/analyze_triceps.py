import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           tricep_torque_share, tricep_activation, penalty=1.0, 
                           is_overhead=False, is_kickback=False, is_pushdown=False, is_compound=False):
    """
    Krzywa oporu dla tricepsów. Ułożenie stawu barkowego determinuje 
    zaangażowanie poszczególnych głów (szczególnie głowy długiej).
    """
    tau = (weight_kg * G * total_moment_arm) * tricep_torque_share
    
    if is_overhead:
        # Ćwiczenia nad głową (Francuz, Overhead Extension): Ekstremalne rozciągnięcie głowy długiej
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.5 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.5 * t_vals # Najciężej na dole, na zablokowanych łokciach opór maleje
    elif is_kickback:
        # Kickbacki z hantlem: Napięcie wynosi ZERA na dole (zwisająca ręka), rośnie do maksa na prostym łokciu
        stretch_bonus_factor = 1.0 
        leverage_factor = np.sin(t_vals * np.pi / 2) # Start z 0 do 1. Brak jakiegokolwiek naciągnięcia
    elif is_pushdown:
        # Ściąganie wyciągu: Stałe, idealne napięcie na głowę boczną i przyśrodkową
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.8 * np.exp(-6 * t_vals))
        # Napięcie szczytowe na dole ruchu dopina podkowę tricepsa
        leverage_factor = 0.8 + 0.2 * np.sin(t_vals * np.pi / 2)
    elif is_compound:
        # Dipy, Wąskie wyciskanie: Wielostawowe, najciężej w fazie maksymalnego rozciągnięcia stawów
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-12 * t_vals))
        leverage_factor = 1.0 - 0.4 * t_vals
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = 1.0 - 0.3 * t_vals

    curve = tau * stretch_bonus_factor * leverage_factor * tricep_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def overhead_extension_kinematics(t, levers, phase="concentric"):
    """Prostowanie ramion nad głową (hantel/kabel): Łokcie pionowo w górze."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.8) / 2
    # Łokcie uniesione nad barki (Y dodatnie)
    elbow_y = L_HUM * 0.90
    elbow_z = -L_HUM * 0.20 # Lekko cofnięte łokcie dla wygody
    
    # Ruch przedramienia od karku (t=0) do góry (t=1)
    angle = np.radians(135) * (1 - t) # Kąt w łokciu schodzi z max zgięcia do 0 (prosta ręka)
    y = elbow_y + L_FOR * np.cos(angle)
    z = elbow_z - L_FOR * np.sin(angle)
    return np.array([x, y, z])

def skullcrusher_kinematics(t, levers, phase="concentric"):
    """Francuz (Lying Triceps Extension): Leżąc, łokcie pionowo w górę lub lekko w stronę głowy."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.9) / 2
    # Dla ciała leżącego grawitacja działa na oś Z (jeśli założyć klasyczne mapowanie). 
    # Tutaj utrzymujemy układ Y jako grawitację dla prostoty (plecy na ziemi to rotacja całego modelu).
    elbow_y = L_HUM * 0.85
    elbow_z = -L_HUM * 0.30 # Łokcie cofnięte za czoło (większe napięcie na głowę długą)
    
    angle = np.radians(120) * (1 - t)
    y = elbow_y + L_FOR * np.cos(angle)
    z = elbow_z - L_FOR * np.sin(angle)
    return np.array([x, y, z])

def pushdown_kinematics(t, levers, phase="concentric"):
    """Prostowanie na wyciągu: Łokcie przytwierdzone do żeber, ruch w dół."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.9) / 2
    # Łokcie ułożone luźno wzdłuż tułowia
    elbow_y = -L_HUM * 0.95
    elbow_z = 0.05
    
    # Od zgięcia na wysokości klatki (t=0) do wyprostu na udach (t=1)
    angle = np.radians(100) * (1 - t)
    y = elbow_y - L_FOR * np.cos(angle)
    z = elbow_z + L_FOR * np.sin(angle)
    return np.array([x, y, z])

def kickback_kinematics(t, levers, phase="concentric"):
    """Dumbbell Kickback: Opad tułowia, ramię równoległe do podłogi (łokieć wysoko w tył)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 1.1) / 2
    # Tułów jest pochylony. Łokieć wysoko w tył (hiperekstensja barku = głowa długa skrócona na maxa)
    elbow_y = 0.0
    elbow_z = -L_HUM * 0.95 
    
    # Ręka zwisa swobodnie pod kątem 90 st (t=0), po czym prostuje się w poziomie (t=1)
    angle = np.radians(90) * (1 - t)
    y = elbow_y - L_FOR * np.sin(angle)
    z = elbow_z - L_FOR * np.cos(angle)
    return np.array([x, y, z])

def close_grip_bench_kinematics(t, levers, phase="concentric"):
    """Wąskie wyciskanie leżąc: Łokcie prowadzone bardzo blisko tułowia."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.8) / 2 # Wąski chwyt
    # Skrócenie dźwigni klatki na rzecz potężnego zgięcia w łokciu
    y = L_HUM * 0.15 + (L_HUM + L_FOR) * 0.80 * t
    z = 0.0
    return np.array([x, y, z])

def triceps_dips_kinematics(t, levers, phase="concentric"):
    """Pompki na poręczach (Pionowe tułowie): Łokcie mocno do tyłu, izolacja tricepsa."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    grip_x = (C_W * 1.1) / 2.0
    # Ciało porusza się pionowo w dół wzdłuż rąk (t=0 to pełne ugięcie)
    y = -L_HUM * 0.90 * t
    z = -L_HUM * 0.40 * (1 - t) # Pionowa sylwetka trzyma nacisk na tricepsie, nie na klatce
    return np.array([grip_x, y, z])

def bench_dips_kinematics(t, levers, phase="concentric"):
    """Pompki odwrotne na ławce: Ręce za plecami, nienaturalne rozciągnięcie torebki stawowej."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 1.0) / 2.0
    # Podparcie za plecami
    y = -L_HUM * 0.85 * t
    z = -L_HUM * 0.60 # Dłonie fizycznie za linią bioder
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_overhead_dumbbell_extension(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "overhead_dumbbell", 25.0)
    weight_per_arm = weight / 2.0 # Zakładając hantel trzymany oburącz lub dwa mniejsze

    total_m_arms = l_forearm * np.sin(np.clip(t_vals + 0.2, 0, 1) * np.pi / 1.5)
    
    # Ogromne naprężenie na głowę długą
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=2.0, tricep_torque_share=0.95, tricep_activation=1.00, is_overhead=True)
    act_tri = min(1.00, raw_score / 170.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"dumbbell", "bench"},
        "biomechanical_bounds": {
            "shoulder_flexion_constant": [160.0, 180.0], # Ręce sztywno nad głową
            "elbow_flexion_bottom": [40.0, 60.0]
        },
        "trajectory_func": overhead_extension_kinematics,
        "act": {"triceps": act_tri, "core": 0.40},
        "fibers": {"triceps_long": 1.00, "triceps_lateral": 0.60, "triceps_medial": 0.50}
    }

def evaluate_skullcrusher(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "skullcrusher", 35.0)
    weight_per_arm = weight / 2.0

    # Przesunięcie łokci do tyłu (za czoło) zmienia profil oporu, dając napięcie również na górze ruchu
    total_m_arms = l_forearm * np.cos(t_vals * np.pi / 2.5)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.5, tricep_torque_share=0.95, tricep_activation=0.95, is_overhead=True)
    act_tri = min(0.98, raw_score / 180.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"ez_bar", "bench"},
        "biomechanical_bounds": {
            "shoulder_flexion_constant": [100.0, 120.0], # Łokcie odchylone lekko do tyłu za pion
            "elbow_flexion_bottom": [50.0, 75.0]
        },
        "trajectory_func": skullcrusher_kinematics,
        "act": {"triceps": act_tri},
        "fibers": {"triceps_long": 0.95, "triceps_lateral": 0.85, "triceps_medial": 0.70}
    }

def evaluate_overhead_cable_extension(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "overhead_cable", 25.0)
    
    total_m_arms = l_forearm * (0.5 + 0.5 * np.sin(t_vals * np.pi))
    
    # Wyciąg z tyłu naciąga triceps w dół i do tyłu, dając niesamowicie gładką krzywą siły na wszystkich głowach
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=2.5, tricep_torque_share=1.00, tricep_activation=1.00, is_overhead=True)
    act_tri = min(1.00, raw_score / 160.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"cable_machine", "rope"},
        "biomechanical_bounds": {
            "shoulder_flexion_constant": [150.0, 175.0],
            "torso_forward_lean": [15.0, 30.0] # Lekkie pochylenie w przód (wypad) dla stabilizacji
        },
        "trajectory_func": overhead_extension_kinematics,
        "act": {"triceps": act_tri, "core": 0.50},
        "fibers": {"triceps_long": 1.00, "triceps_lateral": 0.75, "triceps_medial": 0.65}
    }

def evaluate_dumbbell_kickback(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "kickback_db", 10.0)
    
    # 0 na dole (sin 0), maksimum na górze (sin 90). Fatalny profil biomechaniczny pod kątem hipertrofii rozciągnięcia.
    total_m_arms = l_forearm * np.sin(t_vals * np.pi / 2)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.0, tricep_torque_share=0.80, tricep_activation=0.85, is_kickback=True)
    act_tri = min(0.85, raw_score / 120.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"dumbbell", "bench"},
        "biomechanical_bounds": {
            "shoulder_extension_constant": [170.0, 185.0], # Maksymalne wycofanie łokcia - wyłącza głowę długą
            "elbow_flexion_bottom": [85.0, 100.0]
        },
        "trajectory_func": kickback_kinematics,
        "act": {"triceps": act_tri, "delt_rear": 0.50},
        "fibers": {"triceps_lateral": 1.00, "triceps_medial": 0.95, "triceps_long": 0.10} # Głowa długa nie pracuje z powodu skrajnego skrócenia
    }

def evaluate_cable_pushdown_straight_bar(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "cable_pushdown_bar", 30.0)
    
    total_m_arms = l_forearm * (0.6 + 0.4 * np.sin(t_vals * np.pi))
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.8, tricep_torque_share=0.95, tricep_activation=0.95, is_pushdown=True)
    act_tri = min(0.95, raw_score / 160.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"cable_machine", "straight_bar"},
        "biomechanical_bounds": {
            "elbow_stability": [90.0, 100.0], # Łokcie przyspawane do boków tułowia
            "forearm_pronation": [80.0, 100.0] # Nachwyt wymusza sztywność
        },
        "trajectory_func": pushdown_kinematics,
        "act": {"triceps": act_tri},
        "fibers": {"triceps_lateral": 1.00, "triceps_medial": 0.85, "triceps_long": 0.40}
    }

def evaluate_cable_pushdown_rope(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "cable_pushdown_rope", 25.0)
    
    total_m_arms = l_forearm * (0.5 + 0.5 * np.sin(t_vals * np.pi))
    
    # Sznur pozwala na rozchylenie dłoni na dole (dodatkowe odwodzenie barku w poziomie), co mocniej dopina triceps
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.9, tricep_torque_share=1.00, tricep_activation=0.98, is_pushdown=True)
    act_tri = min(0.98, raw_score / 150.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"cable_machine", "rope"},
        "biomechanical_bounds": {
            "elbow_flare_dynamic": [0.0, 15.0], # Lekkie rozejście się dłoni na zewnątrz
            "forearm_pronation": [40.0, 60.0] # Chwyt neutralny/młotkowy
        },
        "trajectory_func": pushdown_kinematics,
        "act": {"triceps": act_tri},
        "fibers": {"triceps_lateral": 1.00, "triceps_medial": 1.00, "triceps_long": 0.45}
    }

def evaluate_close_grip_bench_press(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    
    weight = _get_weight(prof, "close_grip_bench", 80.0)
    weight_per_arm = weight / 2.0

    # Potężne ugięcie w łokciu skraca ramię siły dla klatki, a wydłuża dla tricepsa
    total_m_arms = l_humerus * np.cos(np.radians(20)) * (1 - 0.7 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.5, tricep_torque_share=0.75, tricep_activation=0.95, is_compound=True)
    act_tri = min(0.95, raw_score / 280.0)

    return {
        "cat": "Push/Arms",
        "subcat": "compound",
        "equipment": {"barbell", "bench", "squat_rack"},
        "biomechanical_bounds": {
            "grip_width_ratio": [0.8, 1.1], # Wąski chwyt (na szerokość barków)
            "elbow_tuck_angle": [10.0, 30.0] # Łokcie bardzo blisko tułowia
        },
        "trajectory_func": close_grip_bench_kinematics,
        "act": {"triceps": act_tri, "chest_mid": 0.60, "delt_front": 0.70},
        "fibers": {"triceps_lateral": 1.00, "triceps_medial": 0.95, "triceps_long": 0.30, "chest_mid": 0.70}
    }

def evaluate_triceps_dips(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "dips_added", 10.0)
    weight_total = body_weight + added_weight
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * np.cos(np.radians(10)) * (1 - 0.5 * t_vals)
    
    # Dipy z pionowym tułowiem maksymalizują moment na stawie łokciowym, eliminując pracę dołu klatki
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.8, tricep_torque_share=0.85, tricep_activation=1.00, is_compound=True)
    act_tri = min(1.00, raw_score / 350.0)

    return {
        "cat": "Push/Arms",
        "subcat": "compound",
        "equipment": {"dip_station"},
        "biomechanical_bounds": {
            "torso_forward_lean": [0.0, 15.0], # Sylwetka wyprostowana pionowo!
            "elbow_flare_angle": [0.0, 20.0]
        },
        "trajectory_func": triceps_dips_kinematics,
        "act": {"triceps": act_tri, "delt_front": 0.60, "chest_lower": 0.40},
        "fibers": {"triceps_lateral": 1.00, "triceps_medial": 1.00, "triceps_long": 0.40, "delt_front": 0.70}
    }

def evaluate_bench_dips(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "bench_dips_added", 20.0)
    # W podporze tyłem nogi spoczywają na ziemi lub ławce, odejmując ok. 35-40% masy ciała
    weight_total = added_weight + body_weight * 0.60
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * np.cos(np.radians(20)) * (1 - 0.6 * t_vals)
    
    # Ze względu na ryzyko uszkodzenia przedniej torebki stawowej barku (skrajny wyprost pod obciążeniem),
    # dodano mały penalty factor
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.2, tricep_torque_share=0.80, tricep_activation=0.85, penalty=0.90, is_compound=True)
    act_tri = min(0.88, raw_score / 250.0)

    return {
        "cat": "Arms",
        "subcat": "compound",
        "equipment": {"bench"},
        "biomechanical_bounds": {
            "shoulder_extension_bottom": [160.0, 185.0], # Niebezpiecznie głębokie wejście stawu barkowego
            "elbow_tuck_angle": [10.0, 30.0]
        },
        "trajectory_func": bench_dips_kinematics,
        "act": {"triceps": act_tri, "delt_front": 0.70},
        "fibers": {"triceps_lateral": 0.90, "triceps_medial": 0.95, "triceps_long": 0.20} # Głowa długa niemal wyłączona z racji barku z tyłu
    }

def evaluate_single_arm_cable_pushdown(prof):
    levers = prof.get("levers", {})
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "single_arm_pushdown", 15.0)
    
    total_m_arms = l_forearm * (0.6 + 0.4 * np.sin(t_vals * np.pi))
    
    # Wykonywanie jednorącz (zarówno podchwytem jak i nachwytem) poprawia skupienie neuro-mięśniowe (activation = 1.00)
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.9, tricep_torque_share=1.00, tricep_activation=1.00, is_pushdown=True)
    act_tri = min(1.00, raw_score / 140.0)

    return {
        "cat": "Arms",
        "subcat": "unilateral_isolation",
        "equipment": {"cable_machine", "single_handle"},
        "biomechanical_bounds": {
            "elbow_stability": [95.0, 100.0],
            "torso_stability": [90.0, 100.0]
        },
        "trajectory_func": pushdown_kinematics,
        "act": {"triceps": act_tri, "core": 0.40},
        "fibers": {"triceps_lateral": 0.95, "triceps_medial": 1.00, "triceps_long": 0.35} # Głowa przyśrodkowa (medial) przejmuje lwią część pracy stabilizującej przy podchwycie
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA TRICEPS
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Overhead_Dumbbell_Extension": evaluate_overhead_dumbbell_extension(user_profile),
        "Skullcrusher": evaluate_skullcrusher(user_profile),
        "Overhead_Cable_Extension": evaluate_overhead_cable_extension(user_profile),
        "Dumbbell_Kickback": evaluate_dumbbell_kickback(user_profile),
        "Cable_Pushdown_Straight_Bar": evaluate_cable_pushdown_straight_bar(user_profile),
        "Cable_Pushdown_Rope": evaluate_cable_pushdown_rope(user_profile),
        "Close_Grip_Bench_Press": evaluate_close_grip_bench_press(user_profile),
        "Triceps_Dips": evaluate_triceps_dips(user_profile),
        "Bench_Dips": evaluate_bench_dips(user_profile),
        "Single_Arm_Cable_Pushdown": evaluate_single_arm_cable_pushdown(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na triceps. Wszystkie metryki kompletne.")
    except ImportError:
        pass