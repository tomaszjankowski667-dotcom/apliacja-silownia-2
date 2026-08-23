import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           abs_torque_share, abs_activation, penalty=1.0, 
                           is_hanging=False, is_reverse_crunch=False, is_isometric_kick=False):
    """
    Krzywa oporu dla dolnego odcinka mięśnia prostego brzucha.
    Kluczem do hipertrofii nie jest zgięcie w biodrze, lecz zgięcie lędźwiowe (tyłopochylenie miednicy).
    """
    tau = (weight_kg * G * total_moment_arm) * abs_torque_share
    
    if is_hanging:
        # Wisząc na drążku: 0 na dole (zwis swobodny), maksimum napięcia gdy nogi są w poziomie, 
        # a pośladki zawijają się w górę (t=1).
        stretch_bonus_factor = 1.0 
        # Leverage factor rośnie dramatycznie w górnej fazie (kiedy miednica rotuje)
        leverage_factor = np.sin(t_vals * np.pi / 2) * (1.0 + 0.5 * t_vals) 
    elif is_reverse_crunch:
        # Odwrotne brzuszki: Skupiają się w 100% na rotacji miednicy.
        # Na ławce ujemnej grawitacja działa najsilniej na samej górze dopięcia.
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.5 * np.exp(-4 * t_vals))
        leverage_factor = 0.5 + 0.8 * t_vals # Najciężej dopiąć miednicę do żeber
    elif is_isometric_kick:
        # Nożyce: Zginacze bioder pracują dynamicznie, brzuch pracuje czysto izometrycznie 
        # (utrzymując miednicę przed opadnięciem w przodopochylenie).
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals) * 1.2 # Stałe, mordercze napięcie izometryczne
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * abs_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def hanging_leg_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy nóg w zwisie na drążku: Unoszenie nóg do poziomu + rotacja miednicy."""
    C_W = levers.get("biacromial_width", 0.41)
    L_FEM = levers.get("L_femur", 0.42)
    L_TIB = levers.get("L_tibia", 0.38)
    L_LEG = L_FEM + L_TIB
    
    x = C_W / 2
    # Oś obrotu to biodra. Zaczynamy ze zwisu (0 st), podnosimy do ok 100 st (lekko powyżej poziomu)
    angle = np.radians(100) * t
    y = -L_LEG * np.cos(angle)
    z = L_LEG * np.sin(angle)
    return np.array([x, y, z])

def captains_chair_kinematics(t, levers, phase="concentric"):
    """Wznosy kolan na poręczach (Kapitańskie krzesło): Zgięte kolana, podparte przedramiona."""
    C_W = levers.get("biacromial_width", 0.41)
    L_FEM = levers.get("L_femur", 0.42)
    
    x = C_W / 2
    # Kolana zgięte, krótsza dźwignia (tylko udo)
    angle = np.radians(110) * t # Ruch kolan wysoko do klatki (wymusza rotację miednicy)
    y = -L_FEM * np.cos(angle)
    z = L_FEM * np.sin(angle)
    return np.array([x, y, z])

def decline_reverse_crunch_kinematics(t, levers, phase="concentric"):
    """Odwrotne brzuszki na ławce skośnej ujemnej: Nogi ugięte, odrywanie pośladków."""
    C_W = levers.get("biacromial_width", 0.41)
    L_FEM = levers.get("L_femur", 0.42)
    
    x = C_W / 2
    # Ciało leży głową w górę (ławka ujemna z perspektywy nóg).
    # Ruch odbywa się "w górę i do tyłu", zwijając biodra.
    angle = np.radians(45) * t # Kąt oderwania i podwinięcia miednicy
    y = L_FEM * np.sin(angle)
    z = L_FEM * np.cos(angle)
    return np.array([x, y, z])

def floor_reverse_crunch_kinematics(t, levers, phase="concentric"):
    """Odwrotne brzuszki na podłodze z dociążeniem na kostkach."""
    C_W = levers.get("biacromial_width", 0.41)
    L_FEM = levers.get("L_femur", 0.42)
    
    x = C_W / 2
    # Poziome leżenie. Zwijanie bioder do klatki w kierunku sufitu (Y dodatnie) i głowy (Z ujemne)
    y = L_FEM * 0.4 * t # Unoszenie bioder
    z = L_FEM * 0.8 * (1 - t) # Przyciąganie kolan z wyprostu do klatki
    return np.array([x, y, z])

def scissor_kicks_kinematics(t, levers, phase="concentric"):
    """Nożyce (Scissor Kicks): Pozycja leżąca, nogi zablokowane tuż nad ziemią."""
    C_W = levers.get("biacromial_width", 0.41)
    L_FEM = levers.get("L_femur", 0.42)
    L_TIB = levers.get("L_tibia", 0.38)
    L_LEG = L_FEM + L_TIB
    
    x = C_W / 2
    # Nogi tuż nad ziemią. T symuluje pojedynczy cykl nożyc jednej nogi.
    y = L_LEG * 0.2 + L_LEG * 0.15 * np.sin(t * np.pi) 
    z = L_LEG * 0.95
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_hanging_leg_raise(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    l_tibia = levers.get("L_tibia", 0.38)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "hanging_leg_raise", 0.0) # Np. hantel między stopami
    # Wznosimy ciężar obu wyprostowanych nóg (około 35% masy ciała) + dociążenie
    weight_legs = (body_weight * 0.35) + added_weight

    # Potężne ramię momentu na poziomie biodra, bo proste nogi (udo + podudzie) ważą swoje
    total_m_arms = (l_femur + l_tibia) * np.sin(t_vals * np.pi / 2)
    
    # Król dolnego brzucha, pod warunkiem, że miednica jest podwijana (zgięcie odcinka lędźwiowego)
    raw_score = calc_raw_physics_score(total_m_arms, weight_legs, rom_bonus=0.5, abs_torque_share=0.85, abs_activation=1.00, is_hanging=True)
    act_lower_abs = min(1.00, raw_score / 350.0)

    return {
        "cat": "Core",
        "subcat": "bodyweight_compound",
        "equipment": {"pull_up_bar"},
        "biomechanical_bounds": {
            "hip_flexion_end": [90.0, 120.0], # Pełen zakres do tyłopochylenia
            "pelvic_posterior_tilt": [15.0, 30.0], # KLUCZ! Bez podwinięcia brzuch nie pracuje
            "torso_stability": [90.0, 100.0] # Brak bujania resztą ciała
        },
        "trajectory_func": hanging_leg_raise_kinematics,
        "act": {"abs_lower": act_lower_abs, "hip_flexors": 0.95, "forearms": 0.60},
        "fibers": {"rectus_abdominis_lower": 1.00, "iliopsoas": 1.00, "rectus_femoris": 0.85}
    }

def evaluate_captains_chair(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)
    
    weight_legs = body_weight * 0.35

    # Zgięte kolana skracają dźwignię o połowę (środek ciężkości przesuwa się na udo)
    total_m_arms = l_femur * np.sin(t_vals * np.pi / 2)
    
    # Mniejsze ramię siły oznacza, że ćwiczenie jest łatwiejsze, idealne do skupienia się na rotacji miednicy
    raw_score = calc_raw_physics_score(total_m_arms, weight_legs, rom_bonus=0.0, abs_torque_share=0.90, abs_activation=0.95, is_hanging=True)
    act_lower_abs = min(0.92, raw_score / 200.0)

    return {
        "cat": "Core",
        "subcat": "bodyweight",
        "equipment": {"captains_chair", "dip_station"},
        "biomechanical_bounds": {
            "hip_flexion_end": [100.0, 130.0], # Kolana muszą wjechać bardzo wysoko (do klatki) by zwinąć miednicę
            "lumbar_spine_flexion": [20.0, 40.0]
        },
        "trajectory_func": captains_chair_kinematics,
        "act": {"abs_lower": act_lower_abs, "hip_flexors": 0.85},
        "fibers": {"rectus_abdominis_lower": 0.95, "iliopsoas": 0.90}
    }

def evaluate_decline_reverse_crunch(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "decline_reverse_crunch", 5.0) # Często hantel między stopami
    weight_legs = (body_weight * 0.35) + added_weight

    # Ławka ujemna sprawia, że szczytowe ugięcie miednicy (które normalnie omija grawitację) 
    # dostaje idealny opór z góry do dołu.
    total_m_arms = l_femur * (0.4 + 0.6 * np.sin(t_vals * np.pi / 2))
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_legs, rom_bonus=1.5, abs_torque_share=1.00, abs_activation=1.00, is_reverse_crunch=True)
    act_lower_abs = min(1.00, raw_score / 220.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"decline_bench", "dumbbell"},
        "biomechanical_bounds": {
            "pelvic_posterior_tilt": [20.0, 45.0], # Maksymalne zrolowanie pośladków
            "hip_flexion_constant": [85.0, 100.0] # Nogi zablokowane w kolanach, unika pracy psoas
        },
        "trajectory_func": decline_reverse_crunch_kinematics,
        "act": {"abs_lower": act_lower_abs, "abs_upper": 0.50},
        "fibers": {"rectus_abdominis_lower": 1.00, "rectus_abdominis_upper": 0.60} # Prawdziwa izolacja dołu
    }

def evaluate_floor_reverse_crunch(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "floor_reverse_crunch", 5.0) # Obciążniki na kostkach (Ankle weights)
    weight_legs = (body_weight * 0.35) + added_weight

    # Na płaskiej ziemi odrywanie bioder jest lżejsze pod kątem fizyki na samym szczycie ruchu
    total_m_arms = l_femur * (0.6 + 0.4 * np.sin(t_vals * np.pi))
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_legs, rom_bonus=0.5, abs_torque_share=0.95, abs_activation=0.90, is_reverse_crunch=True)
    act_lower_abs = min(0.90, raw_score / 180.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"floor_mat", "ankle_weights"},
        "biomechanical_bounds": {
            "pelvic_posterior_tilt": [15.0, 35.0],
            "lumbar_spine_flexion": [20.0, 40.0]
        },
        "trajectory_func": floor_reverse_crunch_kinematics,
        "act": {"abs_lower": act_lower_abs, "obliques": 0.40},
        "fibers": {"rectus_abdominis_lower": 0.95, "transversus_abdominis": 0.60}
    }

def evaluate_scissor_kicks(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    l_tibia = levers.get("L_tibia", 0.38)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "scissor_kicks", 2.0)
    weight_legs = (body_weight * 0.35) + added_weight

    # Proste nogi zawieszone tuż nad ziemią dają absolutnie mordercze, stałe ramię momentu
    total_m_arms = np.full_like(t_vals, (l_femur + l_tibia) * 0.95)
    
    # Nogi pracują dynamicznie, brzuch musi zatrzymać siłę łamiącą kręgosłup (izometria anty-wyprostna)
    raw_score = calc_raw_physics_score(total_m_arms, weight_legs, rom_bonus=0.0, abs_torque_share=0.85, abs_activation=1.00, is_isometric_kick=True)
    act_lower_abs = min(0.95, raw_score / 300.0)

    return {
        "cat": "Core",
        "subcat": "isometric_dynamic", # Brzuch izolacja, biodra dynamika
        "equipment": {"floor_mat", "ankle_weights"},
        "biomechanical_bounds": {
            "lumbar_spine_neutral": [95.0, 100.0], # KLUCZ: lędźwia wklejone w podłogę! Brak wyprostu
            "hip_flexion_dynamic": [10.0, 30.0] # Mały zakres ruchu nożyc
        },
        "trajectory_func": scissor_kicks_kinematics,
        "act": {"abs_lower": act_lower_abs, "hip_flexors": 0.95, "transversus": 0.85},
        "fibers": {"rectus_abdominis_lower": 0.95, "iliopsoas": 1.00, "transversus_abdominis": 0.90}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA DOLNY BRZUCH
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Hanging_Leg_Raise": evaluate_hanging_leg_raise(user_profile),
        "Captains_Chair": evaluate_captains_chair(user_profile),
        "Decline_Reverse_Crunch": evaluate_decline_reverse_crunch(user_profile),
        "Floor_Reverse_Crunch": evaluate_floor_reverse_crunch(user_profile),
        "Scissor_Kicks": evaluate_scissor_kicks(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na dolny odcinek brzucha (Lower Abs). Wszystkie metryki kompletne.")
    except ImportError:
        pass