import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           hamstring_torque_share, hamstring_activation, penalty=1.0, 
                           is_hip_hinge=False, is_nordic=False, is_seated_curl=False):
    """
    Krzywa oporu dla mięśni kulszowo-goleniowych. 
    Uwzględnia różnice między pracą w stawie biodrowym (rozciągnięcie) a kolanowym.
    """
    tau = (weight_kg * G * total_moment_arm) * hamstring_torque_share
    
    # Ruchy zawiasowe (RDL, Good Morning) dają ekstremalny stretch na dole (t=0)
    if is_hip_hinge:
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.8 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.6 * t_vals # Na górze (stojąc) opór na biodrze spada do zera
    elif is_nordic:
        # Żuraw: potężne przeciążenie ekscentryczne na wyprostowanych kolanach (t=0)
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.5 * np.exp(-6 * t_vals))
        leverage_factor = 1.2 - 0.8 * t_vals # Najciężej na samym dole, najlżej w pionie
    else:
        # Maszyny (Leg Curls) mają stały opór dzięki krzywkom
        stretch_bonus_factor = 1.2 if is_seated_curl else 1.0 # Pozycja siedząca wydłuża mięsień na biodrze
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * hamstring_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================
# Dla ćwiczeń tylnej taśmy t=0 to pozycja maksymalnego rozciągnięcia (np. sztanga 
# przy piszczelach w RDL, lub opuszczony tułów w Żurawiu), a t=1 to wyprost.

def rdl_kinematics(t, levers, phase="concentric"):
    """Rumuński Martwy Ciąg (RDL): Biodra idą mocno do tyłu, lekkie ugięcie kolan."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    L_FEM = levers.get("L_femur", 0.42)
    
    x = HIP_W / 2
    # Sztanga opuszcza się pionowo poniżej kolan (t=0), t=1 to postawa wyprostowana
    y = -L_TORSO * 0.85 * (1 - t)
    # Biodra mocno wędrują w tył (Oś Z) w fazie ekscentrycznej
    z = -L_FEM * 0.40 * (1 - t) 
    return np.array([x, y, z])

def stiff_leg_deadlift_kinematics(t, levers, phase="concentric"):
    """Martwy Ciąg na prostych nogach: Sztywniejsze kolana, większy opad tułowia."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    L_FEM = levers.get("L_femur", 0.42)
    
    x = HIP_W / 2
    y = -L_TORSO * 0.95 * (1 - t) # Głębszy opad niż w RDL
    z = -L_FEM * 0.25 * (1 - t) # Biodra idą mniej do tyłu z racji zablokowanych kolan
    return np.array([x, y, z])

def good_morning_kinematics(t, levers, phase="concentric"):
    """Dzień Dobry: Ruch zawiasowy z obciążeniem na karku."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    L_FEM = levers.get("L_femur", 0.42)
    
    x = HIP_W / 2
    # Oś Y i Z dotyczy punktu na karku (gdzie leży sztanga)
    angle = np.radians(80) * (1 - t)
    y = L_TORSO - L_TORSO * np.sin(angle)
    z = -L_FEM * 0.30 * (1 - t) - L_TORSO * np.cos(angle)
    return np.array([x, y, z])

def seated_leg_curl_kinematics(t, levers, phase="concentric"):
    """Uginanie podudzi siedząc: Tułów pochylony pod kątem ok 90st, izolacja kolana."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TIB = levers.get("L_tibia", 0.38)
    
    x = HIP_W / 2
    # t=0 (nogi proste), t=1 (pięty podciągnięte pod uda)
    angle = np.radians(90) * t
    y = -L_TIB * np.sin(angle)
    z = L_TIB * np.cos(angle)
    return np.array([x, y, z])

def lying_leg_curl_kinematics(t, levers, phase="concentric"):
    """Uginanie podudzi leżąc: Biodra w wyproście, izolacja kolana."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TIB = levers.get("L_tibia", 0.38)
    
    x = HIP_W / 2
    # Ławka jest zazwyczaj lekko łamana. Kąt zgięcia kolana: 0 (proste) do ~110 st.
    angle = np.radians(110) * t
    y = L_TIB * np.sin(angle)
    z = -L_TIB * (1 - np.cos(angle))
    return np.array([x, y, z])

def standing_leg_curl_kinematics(t, levers, phase="concentric"):
    """Uginanie podudzi stojąc jednonóż."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TIB = levers.get("L_tibia", 0.38)
    
    x = HIP_W / 2
    angle = np.radians(90) * t
    y = -L_TIB * (1 - np.cos(angle))
    z = -L_TIB * np.sin(angle)
    return np.array([x, y, z])

def nordic_hamstring_curl_kinematics(t, levers, phase="concentric"):
    """Żuraw (Nordic Curl): Kolana zablokowane na ziemi, opad całego tułowia w przód."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    L_TORSO = levers.get("L_torso", 0.50)
    r = L_FEM + L_TORSO # Promień od kolana do głowy
    
    x = HIP_W / 2
    # t=0 (ciało płasko nad ziemią), t=1 (ciało w pionie, klęcząc)
    angle = np.radians(90) * (1 - t)
    y = r * np.cos(angle)
    z = r * np.sin(angle)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_rdl(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    barbell_weight = _get_weight(prof, "rdl_weight", 100.0)
    # W RDL podnosimy sztangę + ciężar górnej połowy ciała (ok. 65% masy ciała)
    weight_total = barbell_weight + (body_weight * 0.65)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_torso * np.cos(np.radians(20)) * (1 - 0.9 * t_vals)
    # RDL mocno angażuje również pośladki.
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.4, hamstring_torque_share=0.65, hamstring_activation=0.95, is_hip_hinge=True)
    act_ham = min(0.95, raw_score / 450.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell", "dumbbells"},
        "biomechanical_bounds": {
            "hip_flexion_bottom": [90.0, 110.0],
            "knee_flexion_constant": [15.0, 30.0] # Lekkie ugięcie kolan odciąża lędźwia
        },
        "trajectory_func": rdl_kinematics,
        "act": {"hamstrings": act_ham, "glutes": 0.85, "lower_back": 0.70},
        "fibers": {"hamstrings_long": 0.95, "hamstrings_short": 0.20, "glute_maximus": 0.90} # Głowa krótka nie pracuje przy zawiasie
    }

def evaluate_stiff_leg_deadlift(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    barbell_weight = _get_weight(prof, "stiff_leg_dl", 90.0)
    weight_total = barbell_weight + (body_weight * 0.65)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_torso * np.cos(np.radians(10)) * (1 - 0.85 * t_vals)
    # Proste nogi zmuszają tył uda do ekstremalnego rozciągnięcia kosztem mniejszego ciężaru
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.8, hamstring_torque_share=0.80, hamstring_activation=1.00, is_hip_hinge=True)
    act_ham = min(1.00, raw_score / 400.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell"},
        "biomechanical_bounds": {
            "hip_flexion_bottom": [95.0, 120.0],
            "knee_flexion_constant": [0.0, 10.0]
        },
        "trajectory_func": stiff_leg_deadlift_kinematics,
        "act": {"hamstrings": act_ham, "glutes": 0.75, "lower_back": 0.85},
        "fibers": {"hamstrings_long": 1.00, "hamstrings_short": 0.25, "glute_maximus": 0.80}
    }

def evaluate_good_morning(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    barbell_weight = _get_weight(prof, "good_morning", 60.0)
    weight_total = barbell_weight + (body_weight * 0.65)
    weight_per_leg = weight_total / 2.0

    # Ramię momentu jest tutaj ogromne, bo sztanga znajduje się na samym końcu tułowia (kark)
    total_m_arms = l_torso * 0.95 * (1 - 0.9 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.5, hamstring_torque_share=0.70, hamstring_activation=0.95, is_hip_hinge=True)
    act_ham = min(0.95, raw_score / 450.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell", "squat_rack"},
        "biomechanical_bounds": {
            "hip_flexion_bottom": [80.0, 100.0],
            "torso_straight_angle": [170.0, 180.0]
        },
        "trajectory_func": good_morning_kinematics,
        "act": {"hamstrings": act_ham, "glutes": 0.80, "lower_back": 0.95},
        "fibers": {"hamstrings_long": 0.95, "hamstrings_short": 0.20, "lower_back": 1.00}
    }

def evaluate_seated_leg_curl(prof):
    levers = prof.get("levers", {})
    l_tibia = levers.get("L_tibia", 0.38)
    
    machine_weight = _get_weight(prof, "seated_leg_curl", 65.0)
    weight_per_leg = machine_weight / 2.0

    total_m_arms = np.full_like(t_vals, l_tibia * 0.85)
    # Siedząc, biodro zgięte pod kątem 90 stopni wydłuża głowę długą kulszowo-goleniowych - potężny sygnał hipertroficzny
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.5, hamstring_torque_share=1.00, hamstring_activation=1.00, is_seated_curl=True)
    act_ham = min(1.00, raw_score / 280.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"leg_curl_machine"},
        "biomechanical_bounds": {
            "knee_flexion_end": [110.0, 130.0],
            "hip_flexion_constant": [85.0, 100.0]
        },
        "trajectory_func": seated_leg_curl_kinematics,
        "act": {"hamstrings": act_ham, "calves": 0.30},
        "fibers": {"hamstrings_long": 1.00, "hamstrings_short": 0.95} # Tu pracuje cała tylna taśma uda
    }

def evaluate_lying_leg_curl(prof):
    levers = prof.get("levers", {})
    l_tibia = levers.get("L_tibia", 0.38)
    
    machine_weight = _get_weight(prof, "lying_leg_curl", 55.0)
    weight_per_leg = machine_weight / 2.0

    total_m_arms = np.full_like(t_vals, l_tibia * 0.85)
    # Leżąc, biodro jest wyprostowane - mniejsze rozciągnięcie głowy długiej niż w wersji siedzącej
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=0.8, hamstring_torque_share=1.00, hamstring_activation=0.90)
    act_ham = min(0.92, raw_score / 280.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"leg_curl_machine"},
        "biomechanical_bounds": {
            "knee_flexion_end": [110.0, 130.0],
            "hip_extension_constant": [160.0, 180.0]
        },
        "trajectory_func": lying_leg_curl_kinematics,
        "act": {"hamstrings": act_ham, "calves": 0.35},
        "fibers": {"hamstrings_long": 0.85, "hamstrings_short": 1.00} # Silniejsza dominacja głowy krótkiej
    }

def evaluate_standing_leg_curl(prof):
    levers = prof.get("levers", {})
    l_tibia = levers.get("L_tibia", 0.38)
    
    machine_weight = _get_weight(prof, "standing_leg_curl", 25.0)
    weight_per_leg = machine_weight

    total_m_arms = np.full_like(t_vals, l_tibia * 0.85)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=0.7, hamstring_torque_share=1.00, hamstring_activation=0.85)
    act_ham = min(0.88, raw_score / 250.0)

    return {
        "cat": "Legs",
        "subcat": "unilateral_isolation",
        "equipment": {"leg_curl_machine", "cable_machine"},
        "biomechanical_bounds": {
            "knee_flexion_end": [110.0, 135.0],
            "hip_extension_constant": [170.0, 180.0]
        },
        "trajectory_func": standing_leg_curl_kinematics,
        "act": {"hamstrings": act_ham, "glutes": 0.20},
        "fibers": {"hamstrings_long": 0.80, "hamstrings_short": 0.95}
    }

def evaluate_nordic_hamstring_curl(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)
    
    # Żuraw operuje masą własną całego korpusu, od kolan w górę
    weight_total = body_weight * 0.70
    weight_per_leg = weight_total / 2.0

    # Największe ramię momentu na świecie, równe długości całego uda i tułowia (dźwignia niemal 1 metr)
    total_m_arms = (l_femur + l_torso) * np.exp(-3 * t_vals)
    # Zdecydowanie najcięższe ćwiczenie w bazie, u większości ludzi aktywacja przebija sufit w ekscentryce
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=2.5, hamstring_torque_share=0.95, hamstring_activation=1.00, is_nordic=True)
    act_ham = min(1.00, raw_score / 400.0)

    return {
        "cat": "Legs",
        "subcat": "bodyweight_compound",
        "equipment": {"bodyweight", "floor_pad"},
        "biomechanical_bounds": {
            "knee_flexion_end": [0.0, 10.0],
            "torso_straight_angle": [170.0, 180.0] # Linia prosta od kolan po barki
        },
        "trajectory_func": nordic_hamstring_curl_kinematics,
        "act": {"hamstrings": act_ham, "glutes": 0.40, "calves": 0.50},
        "fibers": {"hamstrings_long": 1.00, "hamstrings_short": 1.00}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA TYŁ UDA
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "RDL": evaluate_rdl(user_profile),
        "Stiff_Leg_Deadlift": evaluate_stiff_leg_deadlift(user_profile),
        "Good_Morning": evaluate_good_morning(user_profile),
        "Seated_Leg_Curl": evaluate_seated_leg_curl(user_profile),
        "Lying_Leg_Curl": evaluate_lying_leg_curl(user_profile),
        "Standing_Leg_Curl": evaluate_standing_leg_curl(user_profile),
        "Nordic_Hamstring_Curl": evaluate_nordic_hamstring_curl(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na mięśnie kulszowo-goleniowe (tył uda). Wszystkie metryki kompletne.")
    except ImportError:
        pass