import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           adductor_torque_share, adductor_activation, penalty=1.0, 
                           is_machine=False, is_copenhagen=False):
    """
    Krzywa oporu dla przywodzicieli. 
    Uwzględnia potężne naprężenia w płaszczyźnie czołowej (izolacje) oraz strzałkowej (Sumo Squat).
    """
    tau = (weight_kg * G * total_moment_arm) * adductor_torque_share
    
    if is_machine:
        # Maszyna siedząc: gigantyczny stretch na dole (nogi rozszerzone, t=0)
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.0 * np.exp(-8 * t_vals))
        leverage_factor = 0.5 + 0.5 * t_vals # Najciężej na początku ruchu
    elif is_copenhagen:
        # Deska kopenhaska: brutalne napięcie izometryczne/dynamiczne dla dźwigni całego ciała
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.5 * np.exp(-5 * t_vals))
        leverage_factor = np.ones_like(t_vals) * 1.5 # Stały, skrajnie wysoki opór grawitacji
    else:
        # Złożone ruchy wielostawowe (np. Sumo Squat)
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.8 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.4 * t_vals # Klasyczna krzywa przysiadu (ciężej na dole)

    curve = tau * stretch_bonus_factor * leverage_factor * adductor_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def hip_adductor_machine_kinematics(t, levers, phase="concentric"):
    """Maszyna na przywodziciele siedząc: zamykanie nóg do wewnątrz."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    
    # Praca zachodzi na osi X (szerokość). t=0 to maksymalnie rozwarte uda.
    # W miarę postępu (t->1) nogi schodzą się do szerokości bioder.
    x = (HIP_W / 2) + L_FEM * 0.65 * (1 - t)
    y = 0.0 # Brak ruchu pionowego
    z = L_FEM * 0.85 # Długość podparcia ud
    return np.array([x, y, z])

def cable_standing_adduction_kinematics(t, levers, phase="concentric"):
    """Przywodzenie nogi z wyciągiem stojąc: ruch nogi od zewnątrz do wewnątrz, często krzyżując przed osią ciała."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    L_TIB = levers.get("L_tibia", 0.38)
    L_LEG = L_FEM + L_TIB
    
    # Noga zaczyna z odwodzenia (dodatnie X), kończy za osią ciała (ujemne X względem pionu)
    angle_start = np.radians(35)
    angle_end = -np.radians(10)
    current_angle = angle_start * (1 - t) + angle_end * t
    
    x = (HIP_W / 2) + L_LEG * np.sin(current_angle)
    y = L_LEG * (1 - np.cos(current_angle)) # Lekkie unoszenie stopy w górę z powodu łuku
    z = 0.05 * t # Noga lekko mija nogę postawną z przodu
    return np.array([x, y, z])

def sumo_squat_kinematics(t, levers, phase="concentric"):
    """Przysiad Sumo (Sztanga lub Goblet): Szeroka baza wymusza potężną pracę przywodziciela wielkiego."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    
    # Bardzo szeroki rozstaw (znacznie więcej niż zwykły przysiad)
    x = (HIP_W / 2) + L_FEM * 0.45 
    # Zejście bioder w dół (t=0 to dół)
    y = -L_FEM * 0.80 * (1 - t)
    # Biodra idą lekko do tyłu, ale tułów jest znacznie bardziej pionowy niż w RDL/Klasyku
    z = -L_FEM * 0.15 * (1 - t) 
    return np.array([x, y, z])

def copenhagen_plank_kinematics(t, levers, phase="concentric"):
    """Deska Kopenhaska: opuszczenie i uniesienie bioder podczas gdy górna noga trzyma ciężar."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TORSO = levers.get("L_torso", 0.50)
    
    # Ciało leży bokiem. Oś Y symuluje opuszczenie bioder (stretch) i podciągnięcie w górę (skurcz)
    x = HIP_W / 2
    # Opuszczenie miednicy poniżej linii poziomej
    y = -0.15 * (1 - t) 
    z = 0.0 
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_hip_adductor_machine(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    
    machine_weight = _get_weight(prof, "hip_adductor", 65.0)
    weight_per_leg = machine_weight / 2.0

    # Dźwignią jest punkt podparcia padów, zazwyczaj w okolicach kolan (85% długości kości udowej)
    total_m_arms = np.full_like(t_vals, l_femur * 0.85)
    
    # Maksymalna izolacja i rozciągnięcie przywodzicieli (szczególnie długiego i wielkiego)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=2.0, adductor_torque_share=1.00, adductor_activation=1.00, is_machine=True)
    act_add = min(1.00, raw_score / 250.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"adductor_machine"},
        "biomechanical_bounds": {
            "hip_adduction_end": [0.0, 10.0], # Ude stykają się lub prawie się stykają
            "hip_flexion_constant": [85.0, 100.0] # Kąt siedzenia
        },
        "trajectory_func": hip_adductor_machine_kinematics,
        "act": {"adductors": act_add},
        "fibers": {"adductor_magnus": 0.95, "adductor_longus": 1.00, "adductor_brevis": 0.90}
    }

def evaluate_cable_standing_adduction(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    l_tibia = levers.get("L_tibia", 0.38)
    
    cable_weight = _get_weight(prof, "cable_adduction", 20.0)
    
    # Ramię siły równe długości całej nogi (linka często zaczepiona na kostce)
    total_m_arms = np.full_like(t_vals, (l_femur + l_tibia) * 0.9)
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=1.4, adductor_torque_share=1.00, adductor_activation=0.90, is_machine=True)
    act_add = min(0.95, raw_score / 220.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"cable_machine", "ankle_strap"},
        "biomechanical_bounds": {
            "hip_adduction_end": [-15.0, 0.0], # Przekraczanie osi centralnej ciała
            "torso_straight_angle": [170.0, 180.0]
        },
        "trajectory_func": cable_standing_adduction_kinematics,
        "act": {"adductors": act_add},
        "fibers": {"adductor_longus": 1.00, "adductor_magnus": 0.70, "pectineus": 0.85}
    }

def evaluate_sumo_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)
    
    barbell_weight = _get_weight(prof, "sumo_squat", 80.0) # Może to być sztanga, Kettlebell lub Hantel (Goblet)
    weight_total = barbell_weight + (body_weight * 0.88)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_femur * np.cos(np.radians(25)) * (1 - 0.7 * t_vals)
    
    # Z powodu szerokiej bazy i rotacji zewnętrznej, przywodziciel wielki (Adductor Magnus) 
    # działa z potężną mocą jako prostownik biodra u dołu ruchu.
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.5, adductor_torque_share=0.45, adductor_activation=0.95)
    act_add = min(0.95, raw_score / 350.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell", "dumbbell", "kettlebell"},
        "biomechanical_bounds": {
            "stance_width_ratio": [1.6, 2.2], # Szersza baza
            "hip_external_rotation": [25.0, 45.0], # Palce i kolana na zewnątrz
            "torso_forward_lean": [10.0, 25.0] # Bardziej pionowy tułów niż w klasyku
        },
        "trajectory_func": sumo_squat_kinematics,
        "act": {"adductors": act_add, "quads": 0.75, "glutes": 0.65},
        "fibers": {"adductor_magnus": 1.00, "adductor_longus": 0.60, "quads_vastus": 0.80, "glute_maximus": 0.75}
    }

def evaluate_copenhagen_plank(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)
    
    # W desce kopenhaskiej górna noga (przywodziciel) dźwiga ogromną część masy ciała (~60%)
    weight_active_leg = body_weight * 0.60

    # Ogromna dźwignia – punkt podparcia na kostce/łydce, a środek ciężkości w okolicach pępka
    total_m_arms = (l_femur + l_torso * 0.5) * np.ones_like(t_vals)
    
    # Czyste przeciążenie strukturalne (izometria/ruch dynamiczny o małym ROM)
    raw_score = calc_raw_physics_score(total_m_arms, weight_active_leg, rom_bonus=1.5, adductor_torque_share=0.95, adductor_activation=1.00, is_copenhagen=True)
    act_add = min(1.00, raw_score / 300.0)

    return {
        "cat": "Legs",
        "subcat": "bodyweight_isolation",
        "equipment": {"bodyweight", "bench"},
        "biomechanical_bounds": {
            "torso_lateral_stability": [95.0, 100.0], # Ciało musi utrzymać linię w płaszczyźnie czołowej
            "hip_adduction_dynamic": [-15.0, 5.0] # ROM dynamicznego unoszenia bioder
        },
        "trajectory_func": copenhagen_plank_kinematics,
        "act": {"adductors": act_add, "core": 0.85},
        "fibers": {"adductor_magnus": 1.00, "adductor_longus": 0.95, "adductor_brevis": 0.85, "obliques": 0.90}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA PRZYWODZICIELE
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Hip_Adductor_Machine": evaluate_hip_adductor_machine(user_profile),
        "Cable_Standing_Adduction": evaluate_cable_standing_adduction(user_profile),
        "Sumo_Squat": evaluate_sumo_squat(user_profile),
        "Copenhagen_Plank": evaluate_copenhagen_plank(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na mięśnie przywodziciele (wewnętrzna część uda). Wszystkie metryki kompletne.")
    except ImportError:
        pass