import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           calf_torque_share, calf_activation, penalty=1.0, 
                           is_seated=False, is_donkey=False):
    """
    Krzywa oporu dla łydek. Opiera się w dużej mierze na potężnym rozciągnięciu 
    w dolnej fazie ruchu (zgięcie grzbietowe stopy).
    """
    tau = (weight_kg * G * total_moment_arm) * calf_torque_share
    
    # Łydki reagują gigantycznie na stretch-mediated hypertrophy (rozciągnięcie pod obciążeniem na dole t=0)
    if is_donkey:
        # Ośle wspięcia dodatkowo naciągają całą taśmę tylną, co potęguje napięcie w brzuchatym łydki
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.5 * np.exp(-10 * t_vals))
    else:
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.8 * np.exp(-12 * t_vals))

    if is_seated:
        # W siadzie krzywka maszyny często wyrównuje opór, a łydka pracuje w krótszym zakresie
        leverage_factor = np.ones_like(t_vals) * 1.1
    else:
        # Na stojąco najciężej jest na dole, a w peak contraction (t=1) staw skokowy blokuje się biomechanicznie
        leverage_factor = 1.2 - 0.4 * t_vals 

    curve = tau * stretch_bonus_factor * leverage_factor * calf_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================
# Dla łydek t=0 to maksymalne rozciągnięcie (pięta nisko), a t=1 to 
# maksymalne spięcie (pięta wysoko, stanie na palcach).

def standing_calf_raise_kinematics(t, levers, phase="concentric"):
    """Wspięcia stojąc: Czysty ruch pionowy całego ciała, napędzany ze stawu skokowego."""
    HIP_W = levers.get("hip_width", 0.32)
    # Ruch na osi Y to około 10-15 cm (zależnie od mobilności stawu skokowego)
    ROM_Y = 0.12 
    
    x = HIP_W / 2
    # t=0 pięta opuszczona poniżej linii palców, t=1 pełne wspięcie
    y = -ROM_Y * (1 - t)
    z = 0.0 # Brak ruchu przód-tył
    return np.array([x, y, z])

def seated_calf_raise_kinematics(t, levers, phase="concentric"):
    """Wspięcia siedząc: Zgięte kolana pod kątem 90 stopni, ruch kolan w górę."""
    HIP_W = levers.get("hip_width", 0.32)
    ROM_Y = 0.10
    
    x = HIP_W / 2
    y = -ROM_Y * (1 - t)
    # Niewielkie przesunięcie kolana do przodu (Z) podczas unoszenia pięty w siadzie
    z = 0.02 * t 
    return np.array([x, y, z])

def leg_press_calf_raise_kinematics(t, levers, phase="concentric"):
    """Wspięcia na suwnicy: Wektor siły odchylony o 45 stopni."""
    HIP_W = levers.get("hip_width", 0.32)
    ROM = 0.12
    
    x = HIP_W / 2
    # Odpychanie platformy po skosie (Ruch Y i Z połączone przez kąt suwnicy)
    angle = np.radians(45)
    y = ROM * np.sin(angle) * t
    z = ROM * np.cos(angle) * t
    return np.array([x, y, z])

def single_leg_calf_raise_kinematics(t, levers, phase="concentric"):
    """Wspięcia jednonóż: Tożsamy ruch pionowy jak w staniu, ale asymetryczny."""
    HIP_W = levers.get("hip_width", 0.32)
    ROM_Y = 0.12
    
    x = HIP_W / 2
    y = -ROM_Y * (1 - t)
    z = 0.0
    return np.array([x, y, z])

def donkey_calf_raise_kinematics(t, levers, phase="concentric"):
    """Ośle wspięcia: Tułów pochylony do 90 stopni, ciężar na miednicy."""
    HIP_W = levers.get("hip_width", 0.32)
    ROM_Y = 0.14 # Często głębszy stretch niż przy zwykłym staniu
    
    x = HIP_W / 2
    # Pionowy ruch bioder pod wpływem stawu skokowego
    y = -ROM_Y * (1 - t)
    z = 0.01 * t # Znikome bujnięcie ciała
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_standing_calf_raise(prof):
    levers = prof.get("levers", {})
    body_weight = prof.get("weight_kg", 85.0)
    
    machine_weight = _get_weight(prof, "standing_calf", 80.0)
    # Stojąc, łydki muszą podnieść ciężar + 100% masy ciała
    weight_total = machine_weight + body_weight
    weight_per_leg = weight_total / 2.0

    # Ramię momentu na stawie skokowym to odległość od osi obrotu (kostki) do punktu przyłożenia siły (palce)
    # Zwykle to krótka dźwignia, wynosząca około 12-15 cm.
    total_m_arms = np.full_like(t_vals, 0.15)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.5, calf_torque_share=1.00, calf_activation=0.95)
    act_calf = min(0.95, raw_score / 200.0)

    return {
        "cat": "Calves",
        "subcat": "isolation",
        "equipment": {"calf_machine", "barbell", "dumbbells"},
        "biomechanical_bounds": {
            "knee_extension_constant": [170.0, 180.0], # Proste kolana do aktywacji mięśnia brzuchatego
            "ankle_dorsiflexion_bottom": [20.0, 40.0]
        },
        "trajectory_func": standing_calf_raise_kinematics,
        "act": {"calves": act_calf, "core": 0.40},
        "fibers": {"calves_gastrocnemius": 0.95, "calves_soleus": 0.85} # Brzuchaty mocno zaangażowany
    }

def evaluate_seated_calf_raise(prof):
    levers = prof.get("levers", {})
    
    machine_weight = _get_weight(prof, "seated_calf", 50.0)
    # Siedząc, nie podnosimy własnej masy ciała (tylko uda, co można pominąć)
    weight_per_leg = machine_weight / 2.0

    total_m_arms = np.full_like(t_vals, 0.15)
    
    # Siedząc, zgięte kolano skraca mięsień brzuchaty (Gastrocnemius) niemal do zera. 
    # 100% obciążenia przejmuje płaszczkowaty (Soleus).
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.0, calf_torque_share=1.00, calf_activation=1.00, is_seated=True)
    act_calf = min(0.95, raw_score / 140.0)

    return {
        "cat": "Calves",
        "subcat": "isolation",
        "equipment": {"seated_calf_machine"},
        "biomechanical_bounds": {
            "knee_flexion_constant": [80.0, 100.0], # Zgięte kolano wyłącza mięsień brzuchaty
            "ankle_dorsiflexion_bottom": [20.0, 45.0]
        },
        "trajectory_func": seated_calf_raise_kinematics,
        "act": {"calves": act_calf},
        "fibers": {"calves_soleus": 1.00, "calves_gastrocnemius": 0.15} # Izolacja płaszczkowatego
    }

def evaluate_leg_press_calf_raise(prof):
    levers = prof.get("levers", {})
    
    machine_weight = _get_weight(prof, "leg_press_calf", 120.0)
    # Ciężar suwnicy rzutowany pod kątem 45 stopni
    effective_weight = machine_weight * np.cos(np.radians(45))
    weight_per_leg = effective_weight / 2.0

    total_m_arms = np.full_like(t_vals, 0.15)
    
    # Podobne do wspięć stojąc, ale bez obciążania kręgosłupa własną masą ciała
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.4, calf_torque_share=1.00, calf_activation=0.90)
    act_calf = min(0.92, raw_score / 180.0)

    return {
        "cat": "Calves",
        "subcat": "isolation",
        "equipment": {"leg_press_machine"},
        "biomechanical_bounds": {
            "knee_extension_constant": [170.0, 180.0],
            "ankle_dorsiflexion_bottom": [15.0, 35.0]
        },
        "trajectory_func": leg_press_calf_raise_kinematics,
        "act": {"calves": act_calf},
        "fibers": {"calves_gastrocnemius": 0.90, "calves_soleus": 0.85}
    }

def evaluate_single_leg_calf_raise(prof):
    levers = prof.get("levers", {})
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "single_leg_calf", 20.0)
    # Cały ciężar (własny + dodany) spoczywa na jednej nodze
    weight_active_leg = added_weight + body_weight

    total_m_arms = np.full_like(t_vals, 0.15)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_active_leg, rom_bonus=1.5, calf_torque_share=1.00, calf_activation=0.95)
    act_calf = min(1.00, raw_score / 250.0) # Wyższy próg ze względu na duże sumaryczne obciążenie jednej nogi

    return {
        "cat": "Calves",
        "subcat": "unilateral_isolation",
        "equipment": {"dumbbell", "bodyweight"},
        "biomechanical_bounds": {
            "knee_extension_constant": [170.0, 180.0],
            "ankle_dorsiflexion_bottom": [20.0, 40.0]
        },
        "trajectory_func": single_leg_calf_raise_kinematics,
        "act": {"calves": act_calf, "core": 0.30},
        "fibers": {"calves_gastrocnemius": 1.00, "calves_soleus": 0.90}
    }

def evaluate_donkey_calf_raise(prof):
    levers = prof.get("levers", {})
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "donkey_calf", 40.0) # Może to być maszyna lub np. obciążenie na pasie / inna osoba
    weight_total = added_weight + body_weight * 0.70 # Odliczona waga ramion/głowy opartych o maszynę
    weight_per_leg = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, 0.15)
    
    # Ośle wspięcia zapewniają unikalny stretch ze względu na mocne zgięcie bioder 
    # (naciągnięta powięź i taśma tylna wspomagają rozciągnięcie brzuchatego)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=2.0, calf_torque_share=1.00, calf_activation=1.00, is_donkey=True)
    act_calf = min(1.00, raw_score / 220.0)

    return {
        "cat": "Calves",
        "subcat": "compound_isolation", # Izolacja stawu skokowego, ale angażuje stabilizację całego ciała
        "equipment": {"donkey_calf_machine", "weight_belt"},
        "biomechanical_bounds": {
            "knee_extension_constant": [170.0, 180.0],
            "hip_flexion_constant": [80.0, 100.0], # Pochylony tułów - specyfika oślich wspięć
            "ankle_dorsiflexion_bottom": [25.0, 45.0]
        },
        "trajectory_func": donkey_calf_raise_kinematics,
        "act": {"calves": act_calf, "hamstrings": 0.20}, # Lekkie napięcie izometryczne dwugłowych udo/pośladek
        "fibers": {"calves_gastrocnemius": 1.00, "calves_soleus": 0.85}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA ŁYDKI
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Standing_Calf_Raise": evaluate_standing_calf_raise(user_profile),
        "Seated_Calf_Raise": evaluate_seated_calf_raise(user_profile),
        "Leg_Press_Calf_Raise": evaluate_leg_press_calf_raise(user_profile),
        "Single_Leg_Calf_Raise": evaluate_single_leg_calf_raise(user_profile),
        "Donkey_Calf_Raise": evaluate_donkey_calf_raise(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na mięśnie łydek. Wszystkie metryki kompletne.")
    except ImportError:
        pass