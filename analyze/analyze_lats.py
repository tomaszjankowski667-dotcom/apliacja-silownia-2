import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           lat_torque_share, lat_activation, penalty=1.0, 
                           is_vertical_pull=False, is_horizontal_row=False, is_straight_arm=False):
    """
    Krzywa oporu dla mięśnia najszerszego grzbietu.
    Uwzględnia potężny potencjał rozciągnięcia przy ruchach wertykalnych (ściągania/podciągania)
    oraz zmianę profilu oporu przy wiosłowaniach.
    """
    tau = (weight_kg * G * total_moment_arm) * lat_torque_share
    
    if is_vertical_pull:
        # Podciąganie/Ściąganie drążka: Gigantyczny stretch na górze ruchu (t=0). 
        # Z powodu aktywnej niewydolności (active insufficiency), na samym dole (t=1) latsy tracą siłę.
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.0 * np.exp(-10 * t_vals))
        leverage_factor = 1.2 - 0.5 * t_vals # Najciężej dopiąć ruch na samym dole do klatki
    elif is_horizontal_row:
        # Wiosłowania: Główna praca w wyproście stawu ramiennego. 
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-8 * t_vals))
        leverage_factor = 0.8 + 0.4 * np.sin(t_vals * np.pi) # Siła rośnie do środka ruchu, spada na końcu
    elif is_straight_arm:
        # Przenoszenie na prostych rękach (Narciarz): Izolacja wyprostu. Stałe napięcie z kabla.
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-6 * t_vals))
        leverage_factor = 0.6 + 0.4 * np.cos(t_vals * np.pi / 2.5) # Największy opór w pierwszej połowie ruchu
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * lat_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def lat_pulldown_kinematics(t, levers, phase="concentric", grip_width_factor=1.5):
    """Ściąganie drążka wyciągu górnego (szeroko/neutralnie)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * grip_width_factor) / 2
    # Start z maksymalnego wyciągnięcia w górę (t=0), koniec na klatce (t=1)
    y = L_HUM * 0.90 - (L_HUM + L_FOR) * 0.80 * t
    # Drążek omija lekko głowę z przodu i ląduje na górnej klatce
    z = 0.15 * (1 - t) - 0.05 * t 
    return np.array([x, y, z])

def pull_up_kinematics(t, levers, phase="concentric", grip_width_factor=1.4):
    """Podciąganie na drążku: Z punktu widzenia fizyki tułowia ruch jest identyczny z pulldownem, 
       ale angażuje więcej mięśni stabilizujących rdzeń."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * grip_width_factor) / 2
    y = L_HUM * 0.95 - (L_HUM + L_FOR) * 0.85 * t
    z = 0.10 * (1 - t)
    return np.array([x, y, z])

def single_arm_pulldown_kinematics(t, levers, phase="concentric"):
    """Ściąganie jednorącz (często z mocnym rotowaniem tułowia i dopięciem łokcia do boku)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    # Ręka zbiega z zewnątrz w stronę żeber
    x = (C_W * 1.2) / 2 * (1 - t) + (C_W * 0.6) / 2 * t
    y = L_HUM * 0.95 - (L_HUM + L_FOR) * 0.85 * t
    z = 0.10 * (1 - t)
    return np.array([x, y, z])

def dumbbell_row_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie hantlem w opadzie tułowia: Łokieć prowadzony blisko ciała (wyprost)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.9) / 2 # Blisko ciała
    # Tułów poziomo. Hantel ciągnięty w górę przeciw grawitacji.
    y = -(L_HUM + L_FOR) * 0.80 * (1 - t)
    # Ręka wędruje mocno w tył w stronę biodra (kluczowe dla latsów)
    z = 0.15 * (1 - t) - L_HUM * 0.60 * t 
    return np.array([x, y, z])

def seated_cable_row_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie na wyciągu dolnym siedząc: Pionowy tułów, ruch przyciągania do brzucha."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.8) / 2 # Wąski chwyt, łokcie przy ciele
    # Kabel jest ciągnięty horyzontalnie (w osi Z)
    y = -0.15 # Wysokość dolnego brzucha
    # Od wyciągniętych rąk w przód do dłoni przy pępku
    z = (L_HUM + L_FOR) * 0.90 * (1 - t) - 0.10 * t
    return np.array([x, y, z])

def straight_arm_pulldown_kinematics(t, levers, phase="concentric"):
    """Przenoszenie drążka na prostych rękach (Narciarz)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    x = (C_W * 0.9) / 2
    # Ręce u góry z przodu (ok. 120-130 st), łukiem zjeżdżają do ud (0 st)
    angle = np.radians(130) * (1 - t)
    y = -r * 0.20 + r * np.sin(angle)
    z = r * np.cos(angle)
    return np.array([x, y, z])

def machine_row_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie na maszynie (z łokciem przy ciele) np. Hammer Strength."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.8) / 2
    # Lekko po skosie (zależnie od maszyny, zwykle ruch do dołu-tyłu)
    y = L_HUM * 0.20 * (1 - t) - L_HUM * 0.10 * t
    z = (L_HUM + L_FOR) * 0.85 * (1 - t) - L_HUM * 0.20 * t
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_lat_pulldown(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight_total = _get_weight(prof, "lat_pulldown", 60.0)
    weight_per_arm = weight_total / 2.0

    # Ramię momentu rośnie, gdy łokcie zbliżają się do kąta prostego względem kabla (zwykle środek ruchu)
    total_m_arms = l_humerus * np.sin(t_vals * np.pi) * 0.9
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.5, lat_torque_share=0.85, lat_activation=0.95, is_vertical_pull=True)
    act_lat = min(0.95, raw_score / 280.0)

    return {
        "cat": "Back",
        "subcat": "compound",
        "equipment": {"cable_machine", "wide_bar"},
        "biomechanical_bounds": {
            "shoulder_flexion_start": [160.0, 180.0], # Pełne wyciągnięcie rąk nad głowę
            "torso_backward_lean": [10.0, 30.0] # Lekkie odchylenie do tyłu dla optymalnej krzywej dla latsów
        },
        "trajectory_func": lambda t, lev, p="concentric": lat_pulldown_kinematics(t, lev, p, grip_width_factor=1.6),
        "act": {"lats": act_lat, "biceps": 0.50, "traps_lower": 0.60},
        "fibers": {"lats_iliac": 0.80, "lats_lumbar": 0.95, "lats_thoracic": 0.90, "teres_major": 1.00, "biceps_brachii": 0.60}
    }

def evaluate_pull_up(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    
    added_weight = _get_weight(prof, "pull_up_added", 0.0)
    weight_total = body_weight + added_weight
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * np.sin(t_vals * np.pi) * 0.95
    
    # Podciąganie własnego ciała uruchamia ogromne łańcuchy kinematyczne, aktywacja układu nerwowego jest szczytowa
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.5, lat_torque_share=0.80, lat_activation=1.00, is_vertical_pull=True)
    act_lat = min(1.00, raw_score / 350.0)

    return {
        "cat": "Back",
        "subcat": "bodyweight_compound",
        "equipment": {"pull_up_bar", "bodyweight"},
        "biomechanical_bounds": {
            "shoulder_flexion_start": [160.0, 180.0],
            "torso_stability": [90.0, 100.0] # Brak tzw. kippingu (bujania nogami)
        },
        "trajectory_func": pull_up_kinematics,
        "act": {"lats": act_lat, "biceps": 0.60, "core": 0.70},
        "fibers": {"lats_lumbar": 1.00, "lats_thoracic": 0.95, "teres_major": 0.90, "biceps_brachii": 0.75, "rectus_abdominis": 0.80}
    }

def evaluate_single_arm_pulldown(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight = _get_weight(prof, "single_pulldown", 30.0)
    
    total_m_arms = l_humerus * np.sin(t_vals * np.pi) * 0.9

    # Praca jednorącz pozwala na mocniejsze zgięcie boczne tułowia, rewelacyjnie izolując najniższe włókna (iliac)
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=2.0, lat_torque_share=0.95, lat_activation=1.00, is_vertical_pull=True)
    act_lat = min(1.00, raw_score / 160.0)

    return {
        "cat": "Back",
        "subcat": "unilateral_compound",
        "equipment": {"cable_machine", "single_handle"},
        "biomechanical_bounds": {
            "shoulder_flexion_start": [160.0, 180.0],
            "torso_lateral_flexion": [10.0, 25.0] # Celowe ugięcie tułowia do boku
        },
        "trajectory_func": single_arm_pulldown_kinematics,
        "act": {"lats": act_lat, "biceps": 0.40},
        "fibers": {"lats_iliac": 1.00, "lats_lumbar": 0.95, "teres_major": 0.80} # Król dolnego najszerszego
    }

def evaluate_machine_pulldown(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight_total = _get_weight(prof, "machine_pulldown", 70.0)
    weight_per_arm = weight_total / 2.0

    # Krzywka maszyny zapewnia napięcie w całym zakresie, eliminując martwe punkty
    total_m_arms = np.full_like(t_vals, l_humerus * 0.85)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.4, lat_torque_share=0.90, lat_activation=0.95, is_vertical_pull=True)
    act_lat = min(0.95, raw_score / 250.0)

    return {
        "cat": "Back",
        "subcat": "compound",
        "equipment": {"pulldown_machine"},
        "biomechanical_bounds": {
            "shoulder_flexion_start": [150.0, 175.0],
            "torso_stability": [95.0, 100.0] # Pady blokujące uda, idealna stabilność
        },
        "trajectory_func": lambda t, lev, p="concentric": lat_pulldown_kinematics(t, lev, p, grip_width_factor=1.4),
        "act": {"lats": act_lat, "biceps": 0.45},
        "fibers": {"lats_lumbar": 0.95, "lats_thoracic": 0.95, "teres_major": 0.85}
    }

def evaluate_dumbbell_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight = _get_weight(prof, "dumbbell_row", 35.0)

    # Ramię momentu rośnie wraz z ugięciem łokcia do poziomu tułowia
    total_m_arms = l_humerus * np.cos(np.radians(20)) * (0.3 + 0.7 * np.sin(t_vals * np.pi / 2))
    
    # Wiosłowanie z łokciem przy ciele (wyprost stawu ramiennego) świetnie angażuje latsy
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.2, lat_torque_share=0.85, lat_activation=0.95, is_horizontal_row=True)
    act_lat = min(0.95, raw_score / 220.0)

    return {
        "cat": "Back",
        "subcat": "unilateral_compound",
        "equipment": {"dumbbell", "bench"},
        "biomechanical_bounds": {
            "elbow_tuck_angle": [0.0, 20.0], # Klucz: Łokieć MUSI być blisko tułowia dla latsów
            "torso_forward_lean": [75.0, 95.0]
        },
        "trajectory_func": dumbbell_row_kinematics,
        "act": {"lats": act_lat, "rhomboids": 0.50, "biceps": 0.40},
        "fibers": {"lats_lumbar": 1.00, "lats_thoracic": 0.85, "delt_rear": 0.60, "brachialis": 0.50}
    }

def evaluate_seated_cable_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight_total = _get_weight(prof, "seated_cable_row", 60.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * np.sin(t_vals * np.pi) * 0.9
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.5, lat_torque_share=0.85, lat_activation=0.90, is_horizontal_row=True)
    act_lat = min(0.92, raw_score / 230.0)

    return {
        "cat": "Back",
        "subcat": "compound",
        "equipment": {"cable_machine", "v_handle"},
        "biomechanical_bounds": {
            "torso_forward_lean": [0.0, 15.0], # Pionowy tułów z lekkim wychyleniem w przód po stretch
            "elbow_tuck_angle": [10.0, 25.0]
        },
        "trajectory_func": seated_cable_row_kinematics,
        "act": {"lats": act_lat, "rhomboids": 0.70, "traps_mid": 0.65},
        "fibers": {"lats_thoracic": 0.95, "lats_lumbar": 0.85, "rhomboid_major": 0.80, "biceps_brachii": 0.55}
    }

def evaluate_straight_arm_pulldown(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    weight = _get_weight(prof, "straight_arm_pulldown", 25.0)

    # Największe ramię siły w historii fizyki pleców - długa prosta ręka (ramię + przedramię) wędruje przez kąt 120 stopni
    total_m_arms = (l_humerus + l_forearm) * (0.6 + 0.4 * np.cos(t_vals * np.pi / 2.5))
    
    # 100% izolacja najszerszego (bicepsy/plecy górne wyłączone z ruchu). Król pompnięcia latsów.
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.8, lat_torque_share=1.00, lat_activation=1.00, is_straight_arm=True)
    act_lat = min(1.00, raw_score / 150.0)

    return {
        "cat": "Back",
        "subcat": "isolation",
        "equipment": {"cable_machine", "straight_bar_or_rope"},
        "biomechanical_bounds": {
            "elbow_extension_constant": [165.0, 180.0], # Klucz: Łokcie zablokowane
            "torso_forward_lean": [15.0, 35.0]
        },
        "trajectory_func": straight_arm_pulldown_kinematics,
        "act": {"lats": act_lat, "core": 0.50},
        "fibers": {"lats_iliac": 0.90, "lats_lumbar": 1.00, "lats_thoracic": 0.95, "teres_major": 0.90}
    }

def evaluate_machine_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight = _get_weight(prof, "machine_row", 40.0)
    
    # Stabilizacja maszynowa pozwala na maksymalne generowanie mocy bez obciążania rdzenia
    total_m_arms = np.full_like(t_vals, l_humerus * 0.85)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.2, lat_torque_share=0.90, lat_activation=1.00, is_horizontal_row=True)
    act_lat = min(0.98, raw_score / 200.0)

    return {
        "cat": "Back",
        "subcat": "unilateral_compound",
        "equipment": {"rowing_machine", "chest_pad"},
        "biomechanical_bounds": {
            "torso_stability": [98.0, 100.0], # Oparcie klatką piersiową
            "elbow_tuck_angle": [0.0, 20.0] # Dołem przy ciele dla latsów
        },
        "trajectory_func": machine_row_kinematics,
        "act": {"lats": act_lat, "rhomboids": 0.45},
        "fibers": {"lats_lumbar": 1.00, "lats_thoracic": 0.90, "teres_major": 0.80}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA NAJSZERSZY GRZBIETU
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Lat_Pulldown": evaluate_lat_pulldown(user_profile),
        "Pull_Up": evaluate_pull_up(user_profile),
        "Single_Arm_Lat_Pulldown": evaluate_single_arm_pulldown(user_profile),
        "Machine_Lat_Pulldown": evaluate_machine_pulldown(user_profile),
        "Dumbbell_Row": evaluate_dumbbell_row(user_profile),
        "Seated_Cable_Row": evaluate_seated_cable_row(user_profile),
        "Straight_Arm_Pulldown": evaluate_straight_arm_pulldown(user_profile),
        "Machine_Row_Lat_Focus": evaluate_machine_row(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na najszerszy grzbietu. Wszystkie metryki kompletne.")
    except ImportError:
        pass