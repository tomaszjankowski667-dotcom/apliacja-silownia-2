import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           abs_torque_share, abs_activation, penalty=1.0, 
                           is_cable=False, is_decline=False, is_swiss_ball=False, is_floor=False):
    """
    Krzywa oporu dla górnego odcinka mięśnia prostego brzucha.
    Zgięcie kręgosłupa to klucz do aktywacji. Piłka gimnastyczna daje najlepszy stretch, 
    a kabel zapewnia stałe napięcie w całym ruchu.
    """
    tau = (weight_kg * G * total_moment_arm) * abs_torque_share
    
    if is_cable:
        # Allahy na wyciągu: Stałe napięcie gwarantowane przez linkę. Dobry stretch na górze.
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-6 * t_vals))
        leverage_factor = 0.8 + 0.2 * np.sin(t_vals * np.pi) # Równy opór, lekko cięższy w połowie
    elif is_swiss_ball:
        # Piłka szwajcarska: Pozwala na wejście w hiperekstensję (przeprost). Ekstremalny stretch brzucha.
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.5 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.5 * t_vals # Najciężej na dole z rozciągnięcia, lżej po spięciu
    elif is_decline:
        # Ławka ujemna: Potężne ramię momentu na dole (tułów równolegle do podłogi). 
        # Napięcie drastycznie spada, gdy tułów zbliża się do pionu (t=1).
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-8 * t_vals))
        leverage_factor = 1.0 - 0.8 * t_vals 
    elif is_floor:
        # Klasyczne brzuszki na podłodze z ciężarem: Bardzo krótki ROM (tylko łopatki odrywają się od ziemi).
        stretch_bonus_factor = 1.0
        leverage_factor = 1.0 - 0.6 * t_vals # Opór spada w miarę unoszenia barków
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * abs_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def cable_crunch_kinematics(t, levers, phase="concentric"):
    """Allahy (Brzuszki na wyciągu klęcząc): Zwijanie tułowia do kolan."""
    C_W = levers.get("biacromial_width", 0.41)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = C_W / 2
    # Zaczynamy z niemal pionowym tułowiem klęcząc (t=0)
    # Zwijamy się tak, że barki idą w dół i do środka (zgięcie kręgosłupa)
    angle = np.radians(80) * t # Tułów zwija się o ok 80 stopni
    y = L_TORSO * 0.8 * (1 - np.sin(angle)) # Głowa schodzi do podłogi
    z = L_TORSO * 0.5 * np.cos(angle) # Ruch po łuku do tyłu w stronę kolan
    return np.array([x, y, z])

def decline_sit_up_kinematics(t, levers, phase="concentric"):
    """Brzuszki na ławce ujemnej: Unoszenie tułowia z dociążeniem na klatce."""
    C_W = levers.get("biacromial_width", 0.41)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = C_W / 2
    # Tułów leży na skosie (ujemne ułożenie), t=0 to najniższy punkt
    angle_start = -np.radians(30) # Kąt ławki
    angle_end = np.radians(60)    # Uniesienie do siadu (ale nie pełnego pionu, by utrzymać napięcie)
    
    current_angle = angle_start + (angle_end - angle_start) * t
    y = L_TORSO * np.sin(current_angle)
    z = L_TORSO * np.cos(current_angle)
    return np.array([x, y, z])

def machine_crunch_kinematics(t, levers, phase="concentric"):
    """Brzuszki na maszynie siedząc: Wypychanie padu z klatki piersiowej w dół."""
    C_W = levers.get("biacromial_width", 0.41)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = C_W / 2
    # Maszyna ma określoną oś obrotu w pasie.
    angle = np.radians(60) * t
    y = L_TORSO * (1 - np.sin(angle)) # Klatka w dół
    z = L_TORSO * (1 - np.cos(angle)) # I lekko do przodu
    return np.array([x, y, z])

def weighted_crunch_floor_kinematics(t, levers, phase="concentric"):
    """Skłony na podłodze z ciężarem (Crunches): Krótki ruch odrywający łopatki."""
    C_W = levers.get("biacromial_width", 0.41)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = C_W / 2
    # Tylko górna połowa tułowia bierze udział w ruchu
    angle = np.radians(35) * t # Bardzo płytki zakres, ok. 30-40 st
    y = L_TORSO * 0.5 * np.sin(angle)
    z = L_TORSO * 0.5 * np.cos(angle)
    return np.array([x, y, z])

def swiss_ball_crunch_kinematics(t, levers, phase="concentric"):
    """Spięcia na piłce: Przeprost kręgosłupa (stretch) do lekkiego zgięcia."""
    C_W = levers.get("biacromial_width", 0.41)
    L_TORSO = levers.get("L_torso", 0.50)
    
    x = C_W / 2
    # Zaczynamy z plecami wygiętymi na piłce do tyłu (-30 st)
    angle_start = -np.radians(30)
    angle_end = np.radians(30) # Spięcie tylko do pionowego brzucha
    
    current_angle = angle_start + (angle_end - angle_start) * t
    y = L_TORSO * np.sin(current_angle)
    z = L_TORSO * np.cos(current_angle)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_cable_crunch(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    
    weight = _get_weight(prof, "cable_crunch", 40.0) # Zwykle spory ciężar z całego stosu

    # Ramię siły równe długości całego tułowia (ciągnięcie sznura przy głowie)
    total_m_arms = l_torso * (0.7 + 0.3 * np.sin(t_vals * np.pi))
    
    # Allahy (klęcząc) idealnie izolują górny odcinek prostego, gdy utrzymamy biodra w bezruchu
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.5, abs_torque_share=0.95, abs_activation=1.00, is_cable=True)
    act_upper_abs = min(1.00, raw_score / 280.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"cable_machine", "rope"},
        "biomechanical_bounds": {
            "lumbar_spine_flexion": [35.0, 60.0], # Pełny koci grzbiet (zgięcie brzucha)
            "hip_flexion_constant": [85.0, 95.0] # Biodra zablokowane w klęku
        },
        "trajectory_func": cable_crunch_kinematics,
        "act": {"abs_upper": act_upper_abs, "abs_lower": 0.40, "lats": 0.30}, # Najszerszy stabilizuje sznur
        "fibers": {"rectus_abdominis_upper": 1.00, "rectus_abdominis_lower": 0.60, "transversus_abdominis": 0.50}
    }

def evaluate_decline_sit_up(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "decline_sit_up", 15.0) # Talerz przy klatce
    # Podnosimy ok. 65% własnej masy (tułów+głowa) oraz ciężar talerza
    weight_total = added_weight + (body_weight * 0.65)

    # Ramię momentu maksymalne na dole (tułów horyzontalnie), spada drastycznie u góry
    total_m_arms = l_torso * np.cos(np.radians(20)) * (1 - 0.7 * t_vals)
    
    # Ćwiczenie złożone. Gdy robimy pełen siad, zginacze bioder (Psoas) przejmują sporą część pracy.
    # Używamy penalty, jeśli plecy są proste. Przykład zakłada prawidłowe, zrolowane plecy.
    raw_score = calc_raw_physics_score(total_m_arms, weight_total, rom_bonus=1.0, abs_torque_share=0.70, abs_activation=0.90, is_decline=True)
    act_upper_abs = min(0.92, raw_score / 320.0)

    return {
        "cat": "Core",
        "subcat": "compound",
        "equipment": {"decline_bench", "weight_plate"},
        "biomechanical_bounds": {
            "hip_flexion_dynamic": [30.0, 90.0], # Zginacze biodra pracują mocno
            "lumbar_spine_flexion": [20.0, 45.0] # Klatka idzie do miednicy
        },
        "trajectory_func": decline_sit_up_kinematics,
        "act": {"abs_upper": act_upper_abs, "hip_flexors": 0.90, "rectus_femoris": 0.70},
        "fibers": {"rectus_abdominis_upper": 0.95, "iliopsoas": 1.00, "rectus_abdominis_lower": 0.75}
    }

def evaluate_machine_crunch(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    
    weight = _get_weight(prof, "machine_crunch", 50.0)

    # Maszyna gwarantuje stabilizację i równe napięcie
    total_m_arms = np.full_like(t_vals, l_torso * 0.75)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.0, abs_torque_share=0.95, abs_activation=0.95, is_cable=True)
    act_upper_abs = min(0.95, raw_score / 250.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"crunch_machine"},
        "biomechanical_bounds": {
            "lumbar_spine_flexion": [20.0, 45.0],
            "pelvic_stability": [95.0, 100.0]
        },
        "trajectory_func": machine_crunch_kinematics,
        "act": {"abs_upper": act_upper_abs, "abs_lower": 0.50},
        "fibers": {"rectus_abdominis_upper": 1.00, "rectus_abdominis_lower": 0.65}
    }

def evaluate_weighted_crunch(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "weighted_crunch", 20.0)
    # Odrywamy tylko około 30% masy ciała (górna połowa tułowia, barki, głowa)
    weight_total = added_weight + (body_weight * 0.30)

    # Największe ramię siły na samym starcie
    total_m_arms = (l_torso * 0.6) * (1 - 0.5 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_total, rom_bonus=0.0, abs_torque_share=0.95, abs_activation=0.90, is_floor=True)
    act_upper_abs = min(0.88, raw_score / 150.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"floor_mat", "dumbbell", "weight_plate"},
        "biomechanical_bounds": {
            "lumbar_spine_flexion": [15.0, 35.0], # Krótki ROM
            "hip_flexion_constant": [80.0, 100.0] # Nogi ugięte dla odizolowania psoas
        },
        "trajectory_func": weighted_crunch_floor_kinematics,
        "act": {"abs_upper": act_upper_abs},
        "fibers": {"rectus_abdominis_upper": 1.00, "rectus_abdominis_lower": 0.30}
    }

def evaluate_swiss_ball_crunch(prof):
    levers = prof.get("levers", {})
    l_torso = levers.get("L_torso", 0.50)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "swiss_ball_crunch", 15.0)
    weight_total = added_weight + (body_weight * 0.40)

    # Zmiana wektora: ramię siły zaczyna działać zza pleców (hiperekstensja)
    total_m_arms = l_torso * np.cos(np.radians(20)) * (1 - 0.4 * t_vals)
    
    # Król hipertrofii brzucha. Przeprost rozciąga "kratę", a następnie wymusza pełen skurcz.
    raw_score = calc_raw_physics_score(total_m_arms, weight_total, rom_bonus=2.5, abs_torque_share=1.00, abs_activation=1.00, is_swiss_ball=True)
    act_upper_abs = min(1.00, raw_score / 220.0)

    return {
        "cat": "Core",
        "subcat": "isolation",
        "equipment": {"swiss_ball", "dumbbell"},
        "biomechanical_bounds": {
            "lumbar_spine_extension": [15.0, 35.0], # Wygięcie pleców na piłce
            "lumbar_spine_flexion": [15.0, 35.0],
            "torso_stability": [70.0, 85.0] # Wymaga mocnej stabilizacji z rdzenia by nie spaść
        },
        "trajectory_func": swiss_ball_crunch_kinematics,
        "act": {"abs_upper": act_upper_abs, "transversus": 0.85},
        "fibers": {"rectus_abdominis_upper": 1.00, "rectus_abdominis_lower": 0.80, "transversus_abdominis": 0.90}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA GÓRNY BRZUCH
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Cable_Crunches": evaluate_cable_crunch(user_profile),
        "Decline_Sit_Ups": evaluate_decline_sit_up(user_profile),
        "Machine_Crunch": evaluate_machine_crunch(user_profile),
        "Weighted_Crunch": evaluate_weighted_crunch(user_profile),
        "Swiss_Ball_Crunch": evaluate_swiss_ball_crunch(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na górny odcinek brzucha (Upper Abs). Wszystkie metryki kompletne.")
    except ImportError:
        pass