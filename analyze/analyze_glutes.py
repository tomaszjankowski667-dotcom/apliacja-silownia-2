import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           glute_torque_share, glute_activation, penalty=1.0, 
                           is_thrust=False, is_abduction=False, is_cable=False):
    """
    Krzywa oporu dla pośladków. Profil oporu zmienia się drastycznie w zależności od wektora.
    """
    tau = (weight_kg * G * total_moment_arm) * glute_torque_share
    
    if is_thrust:
        # Hip Thrust / Glute Bridge: Najlżej na dole, skrajnie ciężko w pełnym wyproście (t=1)
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.2 * np.exp(-4 * t_vals)) # Słaby bonus za rozciągnięcie
        leverage_factor = 0.3 + 0.7 * t_vals # Rosnące ramię siły
    elif is_cable:
        # Linki wyciągu: Stałe napięcie niezależnie od kąta (Kickbacki)
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-8 * t_vals))
        leverage_factor = np.ones_like(t_vals) * 1.2
    elif is_abduction:
        # Odwodzenia na maszynie / gumy: Największe opory przy maksymalnym rozwarciu (t=1)
        stretch_bonus_factor = 1.0
        leverage_factor = 0.5 + 0.5 * t_vals
    else:
        # Standardowe krzywe (np. Back Extension, Sumo DL)
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.3 * t_vals

    curve = tau * stretch_bonus_factor * leverage_factor * glute_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def hip_thrust_kinematics(t, levers, phase="concentric"):
    """Wznosy bioder ze sztangą: Oparcie o ławkę, potężny ruch pionowy bioder."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = HIP_W / 2
    # Biodra zjeżdżają w dół pod ławkę (t=0) i wychodzą do poziomu (t=1)
    y = -L_TORSO * 0.50 * (1 - t)
    z = L_TORSO * 0.10 * (1 - t) 
    return np.array([x, y, z])

def glute_bridge_kinematics(t, levers, phase="concentric"):
    """Mostki pośladkowe: Plecy płasko na ziemi, mniejszy ROM niż Hip Thrust."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = HIP_W / 2
    # Podłoga fizycznie blokuje głębsze zejście, ROM ograniczony
    y = -L_TORSO * 0.35 * (1 - t)
    z = 0.0 
    return np.array([x, y, z])

def single_leg_hip_thrust_kinematics(t, levers, phase="concentric"):
    """Hip Thrust jednonóż: Asymetria z jedną nogą w powietrzu, ruch identyczny z Hip Thrust."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = HIP_W / 2
    y = -L_TORSO * 0.50 * (1 - t)
    z = L_TORSO * 0.10 * (1 - t) 
    return np.array([x, y, z])

def hip_abductor_machine_kinematics(t, levers, phase="concentric"):
    """Maszyna na odwodziciele siedząc: Rozwieranie ud na zewnątrz."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    
    # Praca zachodzi na osi X (szerokość). t=0 to złączone kolana.
    x = (HIP_W / 2) + L_FEM * 0.55 * t
    y = 0.0 # Brak ruchu pionowego
    z = L_FEM * 0.85 # Odległość podparcia na maszynie
    return np.array([x, y, z])

def cable_kickback_kinematics(t, levers, phase="concentric"):
    """Wymachy nogi w tył z wyciągiem dolnym: Czysty wyprost biodra."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    L_TIB = levers.get("L_tibia", 0.38)
    L_LEG = L_FEM + L_TIB
    
    x = HIP_W / 2
    # Noga wędruje do tyłu (ujemne Z) i lekko w górę (dodatnie Y)
    angle = np.radians(45) * t
    y = L_LEG * (1 - np.cos(angle))
    z = -L_LEG * np.sin(angle)
    return np.array([x, y, z])

def cable_lateral_abduction_kinematics(t, levers, phase="concentric"):
    """Odwodzenie nogi w bok na wyciągu: Atak na pośladkowy średni w płaszczyźnie czołowej."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    L_TIB = levers.get("L_tibia", 0.38)
    L_LEG = L_FEM + L_TIB
    
    # Noga wędruje w bok (szeroko na X)
    angle = np.radians(35) * t
    x = (HIP_W / 2) + L_LEG * np.sin(angle)
    y = L_LEG * (1 - np.cos(angle))
    z = 0.0
    return np.array([x, y, z])

def glute_back_extension_kinematics(t, levers, phase="concentric"):
    """Przeprosty na ławce rzymskiej (Glute-focused): Opad tułowia, zaokrąglone plecy."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = HIP_W / 2
    # t=0 (tułów w dół prostopadle do ziemi), t=1 (poziomo)
    angle = np.radians(90) * (1 - t)
    y = -L_TORSO * np.cos(angle)
    z = L_TORSO * np.sin(angle)
    return np.array([x, y, z])

def sumo_deadlift_kinematics(t, levers, phase="concentric"):
    """Martwy ciąg Sumo: Szeroki rozstaw, rotacja zewnętrzna, potężny napęd z pośladka."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    L_FEM = levers.get("L_femur", 0.42)
    
    # Znacznie szersza baza na osi X
    x = (HIP_W / 2) + L_FEM * 0.40
    # Obniżenie bioder (bardziej pionowy tułów niż w klasycznym MC)
    y = -L_TORSO * 0.75 * (1 - t)
    z = -L_FEM * 0.20 * (1 - t)
    return np.array([x, y, z])

def monster_walk_kinematics(t, levers, phase="concentric"):
    """Spacer farmera z gumą (Monster Walk): Niskie zejście, ciągłe szerokie odwodzenie na boki."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    
    # t symuluje krok do boku - poszerzenie bazy z oporem gumy
    x = (HIP_W / 2) + L_FEM * 0.35 + L_FEM * 0.25 * t
    y = -L_FEM * 0.40 # Stałe obniżenie w półprzysiadzie
    z = 0.0
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_hip_thrust(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    barbell_weight = _get_weight(prof, "hip_thrust", 140.0)
    # W thrustach podnosimy mniejszą część masy ciała niż w przysiadach (ok. 50%)
    weight_total = barbell_weight + (body_weight * 0.50)
    weight_per_leg = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, l_torso * 0.35)
    # Ekstremalny skurcz na górze (is_thrust=True), bardzo niska zależność od dołu pośladka
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=0.8, glute_torque_share=0.90, glute_activation=1.00, is_thrust=True)
    act_glute = min(1.00, raw_score / 350.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell", "bench"},
        "biomechanical_bounds": {
            "hip_extension_top": [170.0, 185.0], # Pełny wyprost i dopięcie pośladków
            "knee_flexion_constant": [80.0, 100.0] # Kąt prosty w piszczeli dla maksymalnej izolacji
        },
        "trajectory_func": hip_thrust_kinematics,
        "act": {"glutes": act_glute, "hamstrings": 0.30, "quads": 0.20},
        "fibers": {"glute_maximus": 1.00, "glute_medius": 0.20, "hamstrings_long": 0.30}
    }

def evaluate_glute_bridge(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    barbell_weight = _get_weight(prof, "glute_bridge", 120.0)
    weight_total = barbell_weight + (body_weight * 0.40)
    weight_per_leg = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, l_torso * 0.25)
    # Krótszy ROM niż Thrust, mniejsza dźwignia
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=0.5, glute_torque_share=0.95, glute_activation=0.95, is_thrust=True)
    act_glute = min(0.95, raw_score / 300.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell", "floor_mat"},
        "biomechanical_bounds": {
            "hip_extension_top": [170.0, 180.0],
            "knee_flexion_constant": [75.0, 95.0]
        },
        "trajectory_func": glute_bridge_kinematics,
        "act": {"glutes": act_glute, "hamstrings": 0.20},
        "fibers": {"glute_maximus": 0.95, "glute_medius": 0.15, "hamstrings_short": 0.10}
    }

def evaluate_single_leg_hip_thrust(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "single_leg_thrust", 40.0)
    # Cały ciężar spoczywa na jednej nodze
    weight_active_leg = added_weight + (body_weight * 0.50)

    total_m_arms = np.full_like(t_vals, l_torso * 0.35)
    raw_score = calc_raw_physics_score(total_m_arms, weight_active_leg, rom_bonus=0.9, glute_torque_share=0.85, glute_activation=0.95, is_thrust=True)
    act_glute = min(0.95, raw_score / 380.0)

    return {
        "cat": "Legs",
        "subcat": "unilateral",
        "equipment": {"dumbbell", "bench"},
        "biomechanical_bounds": {
            "hip_extension_top": [170.0, 185.0],
            "pelvic_stability": [90.0, 100.0] # Wymóg trzymania miednicy w poziomie
        },
        "trajectory_func": single_leg_hip_thrust_kinematics,
        "act": {"glutes": act_glute, "core": 0.70, "hamstrings": 0.40},
        "fibers": {"glute_maximus": 0.95, "glute_medius": 0.65, "core": 0.80} # B-stance mocno dopala średni dla stabilizacji
    }

def evaluate_hip_abductor(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    
    machine_weight = _get_weight(prof, "hip_abductor", 65.0)
    weight_per_leg = machine_weight / 2.0

    total_m_arms = np.full_like(t_vals, l_femur * 0.75)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=0.7, glute_torque_share=1.00, glute_activation=1.00, is_abduction=True)
    act_glute = min(1.00, raw_score / 250.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"abductor_machine"},
        "biomechanical_bounds": {
            "hip_abduction_end": [35.0, 55.0],
            "hip_flexion_constant": [85.0, 100.0]
        },
        "trajectory_func": hip_abductor_machine_kinematics,
        "act": {"glutes": act_glute},
        "fibers": {"glute_medius": 1.00, "glute_minimus": 0.95, "glute_maximus": 0.40} # Króluje górny profil pośladka
    }

def evaluate_cable_kickback(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    l_tibia = levers.get("L_tibia", 0.38)
    
    cable_weight = _get_weight(prof, "cable_kickback", 25.0)
    
    # Ramię siły równe długości całej wyprostowanej nogi
    total_m_arms = np.full_like(t_vals, (l_femur + l_tibia) * 0.85)
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=1.4, glute_torque_share=0.90, glute_activation=1.00, is_cable=True)
    act_glute = min(1.00, raw_score / 280.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"cable_machine", "ankle_strap"},
        "biomechanical_bounds": {
            "hip_extension_end": [165.0, 185.0],
            "lower_back_stability": [95.0, 100.0] # Brak przeprostu odcinka lędźwiowego
        },
        "trajectory_func": cable_kickback_kinematics,
        "act": {"glutes": act_glute, "hamstrings": 0.30},
        "fibers": {"glute_maximus": 1.00, "glute_medius": 0.30}
    }

def evaluate_cable_lateral_abduction(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    
    cable_weight = _get_weight(prof, "cable_abduction", 15.0)
    
    total_m_arms = np.full_like(t_vals, (l_femur) * 0.9)
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=1.2, glute_torque_share=1.00, glute_activation=0.90, is_cable=True)
    act_glute = min(0.95, raw_score / 200.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"cable_machine", "ankle_strap"},
        "biomechanical_bounds": {
            "hip_abduction_end": [30.0, 45.0],
            "torso_straight_angle": [170.0, 180.0]
        },
        "trajectory_func": cable_lateral_abduction_kinematics,
        "act": {"glutes": act_glute},
        "fibers": {"glute_medius": 1.00, "glute_minimus": 0.90, "glute_maximus": 0.20}
    }

def evaluate_glute_back_extension(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "back_extension_added", 15.0)
    weight_total = added_weight + (body_weight * 0.65)
    weight_per_leg = weight_total / 2.0

    # Największe ramię siły w poziomie, przy zablokowanych biodrach
    total_m_arms = l_torso * np.cos(np.radians(20)) * (0.4 + 0.6 * t_vals)
    # Zgarbione plecy (glute-focused) eliminują prostowniki i kierują siłę prosto na pośladki
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.6, glute_torque_share=0.85, glute_activation=0.95)
    act_glute = min(0.95, raw_score / 340.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"roman_chair", "dumbbell"},
        "biomechanical_bounds": {
            "hip_flexion_bottom": [90.0, 110.0],
            "thoracic_spine_flexion": [20.0, 35.0] # Celowe 'koci grzbiet' dla izolacji pośladka
        },
        "trajectory_func": glute_back_extension_kinematics,
        "act": {"glutes": act_glute, "hamstrings": 0.60, "lower_back": 0.40},
        "fibers": {"glute_maximus": 0.95, "hamstrings_long": 0.70, "lower_back": 0.50}
    }

def evaluate_sumo_deadlift(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    barbell_weight = _get_weight(prof, "sumo_dl", 120.0)
    weight_total = barbell_weight + (body_weight * 0.60)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_torso * np.cos(np.radians(35)) * (1 - 0.8 * t_vals)
    # Sumo wymusza rotację zewnętrzną i odwiedzenie, co genialnie uruchamia całą bryłę pośladków (wielki + średni)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.0, glute_torque_share=0.65, glute_activation=0.95)
    act_glute = min(0.95, raw_score / 450.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell"},
        "biomechanical_bounds": {
            "stance_width_ratio": [1.5, 2.2], # Bardzo szeroka baza
            "hip_external_rotation": [30.0, 45.0],
            "torso_forward_lean": [15.0, 35.0] # Pionowszy tułów niż w klasyku
        },
        "trajectory_func": sumo_deadlift_kinematics,
        "act": {"glutes": act_glute, "quads": 0.65, "adductors": 0.85, "lower_back": 0.60},
        "fibers": {"glute_maximus": 0.95, "glute_medius": 0.60, "quads_vastus": 0.70, "adductor_magnus": 1.00}
    }

def evaluate_monster_walk(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    
    band_resistance = _get_weight(prof, "band_resistance", 15.0)
    
    total_m_arms = np.full_like(t_vals, l_femur * 0.85)
    # Stałe napięcie i specyficzne pieczenie mięśnia średniego pośladka
    raw_score = calc_raw_physics_score(total_m_arms, band_resistance, rom_bonus=0.5, glute_torque_share=1.00, glute_activation=1.00, is_cable=True)
    act_glute = min(0.90, raw_score / 180.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"resistance_band"},
        "biomechanical_bounds": {
            "knee_flexion_constant": [30.0, 50.0],
            "hip_abduction_constant": [15.0, 30.0]
        },
        "trajectory_func": monster_walk_kinematics,
        "act": {"glutes": act_glute, "quads": 0.30},
        "fibers": {"glute_medius": 1.00, "glute_minimus": 0.95, "glute_maximus": 0.40}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA POŚLADKI
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Hip_Thrust": evaluate_hip_thrust(user_profile),
        "Glute_Bridge": evaluate_glute_bridge(user_profile),
        "Single_Leg_Hip_Thrust": evaluate_single_leg_hip_thrust(user_profile),
        "Hip_Abductor": evaluate_hip_abductor(user_profile),
        "Cable_Kickbacks": evaluate_cable_kickback(user_profile),
        "Cable_Lateral_Abduction": evaluate_cable_lateral_abduction(user_profile),
        "Glute_Focused_Back_Extension": evaluate_glute_back_extension(user_profile),
        "Sumo_Deadlift": evaluate_sumo_deadlift(user_profile),
        "Monster_Walk": evaluate_monster_walk(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na mięśnie pośladkowe. Wszystkie metryki kompletne.")
    except ImportError:
        pass