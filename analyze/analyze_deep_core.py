import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           core_torque_share, core_activation, penalty=1.0, 
                           is_anti_extension=False, is_anti_lateral=False, is_vacuum=False):
    """
    Krzywa oporu dla głębokiego rdzenia i mięśnia poprzecznego.
    Fizyka opiera się na izometrii i siłach łamiących kręgosłup (siły ścinające).
    """
    tau = (weight_kg * G * total_moment_arm) * core_torque_share
    
    if is_anti_extension:
        # Ab Wheel / Plank: Grawitacja próbuje przeprostować (złamać w dół) kręgosłup lędźwiowy.
        # W kółku ramię siły rośnie drastycznie wraz z wychyleniem w przód.
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.5 * np.exp(-5 * (1 - t_vals))) # Największy stretch na końcu (t=1)
        leverage_factor = 0.2 + 1.8 * (t_vals ** 2) # Wykładniczy wzrost trudności przy odjeździe
    elif is_anti_lateral:
        # Spacer farmera jednorącz: Opór jest stały, a mięśnie skośne i poprzeczny po stronie przeciwnej pracują na 100%.
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.5)
        leverage_factor = np.ones_like(t_vals) * 1.5 
    elif is_vacuum:
        # Próżnia: Brak zewnętrznego ciężaru (weight_kg traktujemy jako wirtualny opór trzewi).
        # Napięcie rośnie w miarę wciągania pępka do kręgosłupa.
        stretch_bonus_factor = 1.0
        leverage_factor = 0.5 + 1.5 * t_vals # Najciężej na maksymalnym wciągnięciu
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * core_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def ab_wheel_kinematics(t, levers, phase="concentric"):
    """Kółko do brzucha: Przejście z klęku do pełnego wyprostu tułowia w poziomie."""
    L_TORSO = levers.get("L_torso", 0.50)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r_arms = L_HUM + L_FOR
    
    # Oś Z to kierunek jazdy w przód
    # Na starcie (t=0) ramiona są pod barkami, kąt tułowia do ziemi to ok 60-70 stopni
    # Na końcu (t=1) tułów i ramiona są w jednej linii poziomej blisko ziemi
    angle_torso = np.radians(70) * (1 - t)
    
    x = 0.0 # Ręce złączone na kółku
    y = L_TORSO * np.sin(angle_torso) + r_arms * 0.2 * (1 - t) # Opadanie w stronę podłogi
    z = L_TORSO * np.cos(angle_torso) + r_arms * t # Odjazd kółka daleko w przód
    return np.array([x, y, z])

def plank_kinematics(t, levers, phase="concentric"):
    """Deska (Plank): Czysta izometria, ciało równoległe do ziemi."""
    C_W = levers.get("biacromial_width", 0.41)
    
    x = C_W / 2 # Oparcie na przedramionach
    y = 0.25 # Stała wysokość nad ziemią
    z = 0.0 
    # Brak ruchu w osi czasu t, czyste napięcie trzymające linię
    return np.array([x, y, z])

def suitcase_carry_kinematics(t, levers, phase="concentric"):
    """Spacer farmera jednorącz (Suitcase Carry): Asymetryczne obciążenie ściągające w bok."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 1.3) / 2 # Hantel/Kettlebell trzymany po jednej stronie
    y = -(L_HUM + L_FOR) # Ręka prosta w dół
    z = 0.0
    return np.array([x, y, z])

def stomach_vacuum_kinematics(t, levers, phase="concentric"):
    """Próżnia brzuszna: Ruch tylko wewnątrz jamy brzusznej (wciąganie ściany przedniej)."""
    # Symulujemy wektor Y jako wciąganie brzucha w stronę kręgosłupa
    x = 0.0
    y = 0.15 * t # Głębokość wciągnięcia pępka
    z = 0.0
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_ab_wheel(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    
    # Kółko zmusza rdzeń do utrzymania w górze około 60-70% masy ciała 
    # (reszta opiera się na kolanach i toczy na kółku)
    weight_core = body_weight * 0.65

    # Ramię momentu staje się absurdalnie długie w pełnym wyproście (od kolan aż po dłonie)
    total_m_arms = (l_torso + l_humerus) * (0.2 + 0.8 * t_vals)
    
    # Król ćwiczeń na rdzeń. Zastępuje pracę anty-wyprostną, niszcząc mięsień poprzeczny i prosty.
    raw_score = calc_raw_physics_score(total_m_arms, weight_core, rom_bonus=2.5, core_torque_share=0.95, core_activation=1.00, is_anti_extension=True)
    act_core = min(1.00, raw_score / 350.0)

    return {
        "cat": "Core",
        "subcat": "bodyweight_compound",
        "equipment": {"ab_wheel", "floor_mat"},
        "biomechanical_bounds": {
            "lumbar_spine_neutral": [95.0, 100.0], # Krytyczne: lędźwia nie mogą "wisieć"
            "pelvic_posterior_tilt": [10.0, 30.0] # Lekkie podwinięcie miednicy dla ochrony lędźwi
        },
        "trajectory_func": ab_wheel_kinematics,
        "act": {"transversus": act_core, "abs_upper": 0.95, "abs_lower": 0.90, "lats": 0.60},
        "fibers": {"transversus_abdominis": 1.00, "rectus_abdominis_lower": 0.95, "rectus_abdominis_upper": 1.00, "lats_lower": 0.70}
    }

def evaluate_weighted_plank(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "plank_weight", 20.0) # Talerz na plecach
    weight_core = (body_weight * 0.60) + added_weight

    # Ramię momentu jest w miarę stałe i zlokalizowane pośrodku ciała
    total_m_arms = np.full_like(t_vals, l_torso * 0.5)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_core, rom_bonus=0.0, core_torque_share=1.00, core_activation=1.00, is_anti_extension=True)
    # Znacznie mniejszy dzielnik przez wzgląd na czystą izometrię bez t_vals rosnącego w odjeździe
    act_core = min(0.95, raw_score / 180.0)

    return {
        "cat": "Core",
        "subcat": "isometric_isolation",
        "equipment": {"weight_plate", "floor_mat"},
        "biomechanical_bounds": {
            "lumbar_spine_neutral": [95.0, 100.0],
            "scapular_protraction": [80.0, 100.0] # Zębate przednie trzymają barki
        },
        "trajectory_func": plank_kinematics,
        "act": {"transversus": act_core, "abs_upper": 0.85, "abs_lower": 0.80},
        "fibers": {"transversus_abdominis": 1.00, "rectus_abdominis_upper": 0.90, "serratus_anterior": 0.80}
    }

def evaluate_suitcase_carry(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    c_w = levers.get("biacromial_width", 0.41)
    
    weight = _get_weight(prof, "suitcase_weight", 32.0) # Jeden ciężki kettlebell

    # Siła łamiąca kręgosłup na bok to połowa szerokości barków względem osi centralnej
    total_m_arms = np.full_like(t_vals, c_w * 0.8) 
    
    # Wybitne ćwiczenie asymetryczne dla mięśnia poprzecznego i czworobocznego lędźwi (QL)
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.0, core_torque_share=1.00, core_activation=1.00, is_anti_lateral=True)
    act_core = min(1.00, raw_score / 120.0)

    return {
        "cat": "Core",
        "subcat": "unilateral_isometric",
        "equipment": {"kettlebell", "dumbbell"},
        "biomechanical_bounds": {
            "lateral_flexion_neutral": [95.0, 100.0], # Tułów MUSI być idealnie w pionie, mimo że ciągnie w dół
            "shoulder_depression": [90.0, 100.0]
        },
        "trajectory_func": suitcase_carry_kinematics,
        "act": {"obliques": act_core, "transversus": 0.95, "quadratus_lumborum": 1.00, "forearms": 0.80},
        "fibers": {"internal_oblique": 1.00, "external_oblique": 0.90, "transversus_abdominis": 1.00, "quadratus_lumborum": 1.00}
    }

def evaluate_stomach_vacuum(prof):
    # Próżnia opiera się wyłącznie na skurczu izometrycznym mięśnia poprzecznego.
    # Wirtualny ciężar dla matematyki silnika (opór trzewi i ciśnienia).
    virtual_resistance = 25.0 
    
    total_m_arms = np.full_like(t_vals, 0.20) # Odległość ściany brzucha od kręgosłupa
    
    raw_score = calc_raw_physics_score(total_m_arms, virtual_resistance, rom_bonus=0.0, core_torque_share=1.00, core_activation=1.00, is_vacuum=True)
    act_core = min(1.00, raw_score / 35.0)

    return {
        "cat": "Core",
        "subcat": "internal_isometric",
        "equipment": {"bodyweight"},
        "biomechanical_bounds": {
            "diaphragm_elevation": [90.0, 100.0], # Przepona zasysa w górę
            "transverse_abdominal_compression": [95.0, 100.0] # Pępek do kręgosłupa
        },
        "trajectory_func": stomach_vacuum_kinematics,
        "act": {"transversus": act_core, "pelvic_floor": 0.70},
        "fibers": {"transversus_abdominis": 1.00, "diaphragm": 0.80, "pelvic_floor_muscles": 0.75}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA GŁĘBOKI RDZEŃ (CORE)
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Ab_Wheel": evaluate_ab_wheel(user_profile),
        "Weighted_Plank": evaluate_weighted_plank(user_profile),
        "Suitcase_Carry": evaluate_suitcase_carry(user_profile),
        "Stomach_Vacuum": evaluate_stomach_vacuum(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na mięsień poprzeczny i głęboki core. Wszystkie metryki kompletne.")
    except ImportError:
        pass