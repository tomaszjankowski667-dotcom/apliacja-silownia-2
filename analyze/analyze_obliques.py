import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           oblique_torque_share, oblique_activation, penalty=1.0, 
                           is_rotation=False, is_lateral_flexion=False, is_wiper=False):
    """
    Krzywa oporu dla mięśni skośnych brzucha.
    Uwzględnia różnice między zgięciem bocznym (Lateral Flexion) a rotacją tułowia (Rotation).
    """
    tau = (weight_kg * G * total_moment_arm) * oblique_torque_share
    
    if is_rotation:
        # Woodchoppers, Russian Twist: Napięcie w płaszczyźnie poprzecznej
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-6 * t_vals))
        leverage_factor = 0.8 + 0.4 * np.sin(t_vals * np.pi) # Płynny opór przez cały zakres rotacji
    elif is_lateral_flexion:
        # Skłony boczne: Ramię siły 0 na dole (przy pionowym tułowiu), rośnie przy wychyleniu
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-8 * t_vals))
        leverage_factor = np.sin(np.clip(t_vals + 0.2, 0, 1) * np.pi / 2)
    elif is_wiper:
        # Windshield Wipers: Ekstremalnie długa dźwignia nóg wykręcająca miednicę
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.0 * np.exp(-5 * t_vals))
        leverage_factor = 0.5 + 1.0 * t_vals # Najciężej na samym dole (kiedy nogi są opuszczone bokiem do ziemi)
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * oblique_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def woodchopper_kinematics(t, levers, phase="concentric"):
    """Rotacje tułowia na wyciągu: Ruch rotacyjny tułowia i ramion po skosie w dół lub w poziomie."""
    C_W = levers.get("biacromial_width", 0.41)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = C_W / 2
    # Rotacja tułowia (t=0 - skręcony w stronę wyciągu, t=1 - odwrócony)
    angle = np.radians(120) * t # Pełen zakres to ok 120 stopni rotacji
    y = L_TORSO * 0.8 * (1 - t) # Dla wersji z góry na dół (High-to-Low)
    z = L_TORSO * np.cos(angle)
    return np.array([x, y, z])

def russian_twist_kinematics(t, levers, phase="concentric"):
    """Russian Twist: Siedząc w odchyleniu (izometria prostego), skręty z talerzem z boku na bok."""
    L_TORSO = levers.get("L_torso", 0.50)
    
    # Odchylenie do tyłu (ujemne Z i obniżone Y)
    y = -L_TORSO * 0.5 
    
    # Ruch obciążenia z jednego boku na drugi (t modeluje połowę ruchu: od środka do max skrętu)
    angle = np.radians(80) * t 
    x = L_TORSO * 0.6 * np.sin(angle)
    z = -L_TORSO * 0.4 + L_TORSO * 0.4 * np.cos(angle)
    return np.array([x, y, z])

def side_bend_kinematics(t, levers, phase="concentric"):
    """Skłony boczne: Czyste zgięcie boczne (Lateral Flexion) kręgosłupa."""
    L_TORSO = levers.get("L_torso", 0.50)
    
    # Zgięcie tylko w osi X i Y
    angle = np.radians(40) * t # Skłon boczny do ok 40 stopni
    x = L_TORSO * np.sin(angle)
    y = L_TORSO * (1 - np.cos(angle))
    z = 0.0 # Brak rotacji (w idealnej formie)
    return np.array([x, y, z])

def oblique_crunch_kinematics(t, levers, phase="concentric"):
    """Spięcia z rotacją (Oblique Crunches): Zgięcie w przód i rotacja do przeciwnego kolana."""
    L_TORSO = levers.get("L_torso", 0.50)
    
    angle_flexion = np.radians(35) * t # Zgięcie brzucha w przód
    angle_rotation = np.radians(45) * t # Rotacja
    
    x = L_TORSO * 0.5 * np.sin(angle_rotation)
    y = L_TORSO * 0.5 * np.sin(angle_flexion)
    z = L_TORSO * 0.5 * np.cos(angle_flexion)
    return np.array([x, y, z])

def windshield_wipers_kinematics(t, levers, phase="concentric"):
    """Windshield Wipers (Wycieraczki): Nogi połączone prostowane w górze, opadają na boki."""
    L_FEM = levers.get("L_femur", 0.42)
    L_TIB = levers.get("L_tibia", 0.38)
    L_LEG = L_FEM + L_TIB
    
    # Z pozycji pionowej (środek) nogi opadają w lewo lub prawo
    angle = np.radians(80) * t # Prawie do samej podłogi z boku
    x = L_LEG * np.sin(angle)
    y = -L_LEG * 0.2 * t # Niewielki opad bioder na boku
    z = L_LEG * np.cos(angle)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_woodchoppers(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    
    weight = _get_weight(prof, "woodchopper_weight", 20.0)

    # Ramię momentu rośnie i maleje w zależności od kąta liny względem tułowia
    total_m_arms = l_torso * (0.6 + 0.4 * np.sin(t_vals * np.pi))
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.5, oblique_torque_share=0.95, oblique_activation=0.95, is_rotation=True)
    act_oblique = min(0.95, raw_score / 200.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"cable_machine", "d_handle"},
        "biomechanical_bounds": {
            "torso_rotation": [90.0, 130.0], # Pełen skręt
            "hip_stability": [80.0, 95.0] # Biodra stabilnie, skręca się głównie klatka piersiowa
        },
        "trajectory_func": woodchopper_kinematics,
        "act": {"obliques": act_oblique, "abs_upper": 0.40},
        "fibers": {"external_oblique": 1.00, "internal_oblique": 0.95, "transversus_abdominis": 0.70}
    }

def evaluate_russian_twist(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "russian_twist_weight", 15.0)
    # Górna część tułowia (35%) trzymana izometrycznie + ciężar w rękach
    weight_total = added_weight + (body_weight * 0.35)

    total_m_arms = l_torso * 0.75 * (0.8 + 0.2 * np.sin(t_vals * np.pi))
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_total, rom_bonus=0.5, oblique_torque_share=0.85, oblique_activation=0.90, is_rotation=True)
    act_oblique = min(0.92, raw_score / 250.0)

    return {
        "cat": "Core",
        "subcat": "compound_isolation",
        "equipment": {"weight_plate", "dumbbell", "floor_mat"},
        "biomechanical_bounds": {
            "torso_backward_lean": [35.0, 50.0], # Odchylenie do tyłu trzymające napięcie prostego brzucha
            "torso_rotation": [45.0, 90.0]
        },
        "trajectory_func": russian_twist_kinematics,
        "act": {"obliques": act_oblique, "abs_upper": 0.85, "abs_lower": 0.70},
        "fibers": {"external_oblique": 0.90, "internal_oblique": 0.90, "rectus_abdominis_upper": 0.95}
    }

def evaluate_side_bends(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    
    weight = _get_weight(prof, "side_bend_weight", 25.0)

    # Maksymalne ramię przy największym wychyleniu w bok
    total_m_arms = l_torso * np.sin(np.clip(t_vals + 0.2, 0, 1) * np.pi / 2)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.2, oblique_torque_share=1.00, oblique_activation=0.95, is_lateral_flexion=True)
    act_oblique = min(0.95, raw_score / 180.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"dumbbell", "roman_chair"},
        "biomechanical_bounds": {
            "lateral_flexion": [30.0, 45.0],
            "lumbar_spine_neutral": [95.0, 100.0] # Brak zginania w przód i w tył! Czysty bok.
        },
        "trajectory_func": side_bend_kinematics,
        "act": {"obliques": act_oblique, "quadratus_lumborum": 0.80},
        "fibers": {"external_oblique": 1.00, "internal_oblique": 0.85, "quadratus_lumborum": 0.95}
    }

def evaluate_oblique_crunches(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    # Krótki ruch odrywający bark, obciążenie to część masy ciała
    weight = body_weight * 0.25 

    total_m_arms = l_torso * 0.5 * (1 - 0.5 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.0, oblique_torque_share=0.80, oblique_activation=0.90)
    act_oblique = min(0.88, raw_score / 120.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"floor_mat"},
        "biomechanical_bounds": {
            "torso_rotation": [30.0, 50.0],
            "lumbar_spine_flexion": [15.0, 30.0]
        },
        "trajectory_func": oblique_crunch_kinematics,
        "act": {"obliques": act_oblique, "abs_upper": 0.80},
        "fibers": {"external_oblique": 0.90, "rectus_abdominis_upper": 0.85}
    }

def evaluate_windshield_wipers(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    l_tibia = levers.get("L_tibia", 0.38)
    body_weight = prof.get("weight_kg", 85.0)
    
    weight_legs = body_weight * 0.35 # Ważą proste nogi

    # Dźwignia to cała długość nóg odchylona w bok - gigantyczny moment skręcający kręgosłup
    total_m_arms = (l_femur + l_tibia) * np.sin(np.clip(t_vals + 0.1, 0, 1) * np.pi / 2)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_legs, rom_bonus=1.5, oblique_torque_share=0.95, oblique_activation=1.00, is_wiper=True)
    act_oblique = min(1.00, raw_score / 280.0)

    return {
        "cat": "Core",
        "subcat": "bodyweight_compound",
        "equipment": {"pull_up_bar", "floor_mat"},
        "biomechanical_bounds": {
            "torso_rotation": [70.0, 90.0],
            "hip_flexion_constant": [85.0, 95.0] # Nogi zablokowane pod kątem 90 stopni (prostopadle do tułowia)
        },
        "trajectory_func": windshield_wipers_kinematics,
        "act": {"obliques": act_oblique, "abs_lower": 0.85, "core": 0.90},
        "fibers": {"external_oblique": 1.00, "internal_oblique": 1.00, "transversus_abdominis": 0.95, "rectus_abdominis_lower": 0.85}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA MIĘŚNIE SKOŚNE
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Woodchoppers": evaluate_woodchoppers(user_profile),
        "Russian_Twist": evaluate_russian_twist(user_profile),
        "Side_Bends": evaluate_side_bends(user_profile),
        "Oblique_Crunches": evaluate_oblique_crunches(user_profile),
        "Windshield_Wipers": evaluate_windshield_wipers(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na mięśnie skośne brzucha. Wszystkie metryki kompletne.")
    except ImportError:
        pass