import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81


def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))


def calc_raw_physics_score(total_moment_arm, weight_per_arm_kg, rom_bonus,
                           forearm_torque_share, forearm_activation, penalty=1.0,
                           is_isometric=False, is_roller=False, is_cable=False):
    """
    Krzywa oporu dla przedramion. Z powodu ekstremalnie krótkich dźwigni (tylko dłoń),
    fizyka opiera się na ciągłości napięcia i sile ścisku.
    """
    tau = (weight_kg * G * total_moment_arm) * forearm_torque_share

    if is_isometric:
        # Chwyt szczypcowy, Ściskacze: Całkowity brak ruchu (t_vals nie ma znaczenia). Stałe, miażdżące napięcie.
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals) * 1.5
    elif is_roller:
        # Nawijarka (Wrist Roller): Jedyny ruch, w którym faza koncentryczna płynnie przechodzi w ekscentryczną bez utraty 1% napięcia.
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals) * 1.3
    elif is_cable:
        # Wyciąg daje napięcie nawet w dolnej (startowej) fazie ugięcia
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.0 * np.exp(-6 * t_vals))
        leverage_factor = 0.6 + 0.4 * np.sin(t_vals * np.pi / 2)
    else:
        # Klasyczne zginanie/prostowanie ze sztangą
        stretch_bonus_factor = 1.0 + (rom_bonus * 0.8 * np.exp(-10 * t_vals))
        leverage_factor = np.sin(np.clip(t_vals + 0.2, 0, 1) * np.pi / 2)

    curve = tau * stretch_bonus_factor * leverage_factor * forearm_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def supported_wrist_curl_kinematics(t, levers, phase="concentric", is_reverse=False):
    """Uginanie/Prostowanie nadgarstków w oparciu o ławkę."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HAND = 0.19  # Średnia długość dłoni (kluczowa dla przedramion)

    x = (C_W * 0.8) / 2
    # Przedramiona opierają się o ławkę (stała wysokość Y)
    wrist_y = -0.40
    wrist_z = 0.30

    # Kąt nadgarstka zmienia się o około 120-140 stopni
    angle_start = -np.radians(60) if not is_reverse else -np.radians(50)
    angle_end = np.radians(70) if not is_reverse else np.radians(60)

    current_angle = angle_start + (angle_end - angle_start) * t

    y = wrist_y + L_HAND * 0.5 * np.sin(current_angle)
    z = wrist_z + L_HAND * 0.5 * np.cos(current_angle)
    return np.array([x, y, z])


def behind_back_wrist_curl_kinematics(t, levers, phase="concentric"):
    """Uginanie nadgarstków z tyłu pleców: Sztanga za pośladkami."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    L_HAND = 0.19

    x = (C_W * 1.1) / 2
    # Ręce z tyłu (Z ujemne), wyprostowane
    wrist_y = -(L_HUM + L_FOR)
    wrist_z = -0.15

    angle = -np.radians(40) + np.radians(80) * t
    y = wrist_y + L_HAND * 0.5 * np.sin(angle)
    z = wrist_z - L_HAND * 0.5 * (1 - np.cos(angle))
    return np.array([x, y, z])


def wrist_roller_kinematics(t, levers, phase="concentric"):
    """Nawijarka (Wrist Roller): Ręce wyciągnięte przed siebie, izometryczne trzymanie barków."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 0.8) / 2
    # Ręce uniesione w linii barków (Y ok. 0)
    wrist_y = 0.0
    wrist_z = L_HUM + L_FOR

    # Ruch nadgarstka to mikrorotacje, z punktu widzenia drążka jest to stałe ułożenie
    return np.array([x, wrist_y, wrist_z])


def gripper_kinematics(t, levers, phase="concentric"):
    """Ściskacze dłoni: Zaciśnięcie pięści, dłoń w spoczynku wzdłuż tułowia lub lekko zgięta."""
    C_W = levers.get("biacromial_width", 0.41)

    x = (C_W * 1.2) / 2
    y = -0.60
    z = 0.0
    # Ruch odbywa się tylko w palcach, punkt nadgarstka stoi w miejscu
    return np.array([x, y, z])


def pinch_grip_farmers_walk_kinematics(t, levers, phase="concentric"):
    """Spacer farmera szczypcowy: Ręce prosto w dół, trzymają talerze obciążeniowe opuszkiem palców."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)

    x = (C_W * 1.3) / 2
    y = -(L_HUM + L_FOR)
    z = 0.0
    # Całkowita izometria w osi X, Y, Z
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_reverse_wrist_curl(prof):
    levers = prof.get("levers", {})
    l_hand = 0.19

    weight = _get_weight(prof, "reverse_wrist_curl", 15.0)
    weight_per_arm = weight / 2.0

    # Ramię siły równe połowie długości dłoni (od nadgarstka do sztangi)
    total_m_arms = (l_hand * 0.5) * np.sin(np.clip(t_vals + 0.2, 0, 1) * np.pi / 2)

    # Niskie obciążenia fizyczne (kilogramy), więc dzielnik raw_score jest drastycznie obniżony (do 15-20)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.0, forearm_torque_share=0.95,
                                       forearm_activation=0.95)
    act_forearm = min(0.95, raw_score / 18.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"barbell", "dumbbells", "bench"},
        "biomechanical_bounds": {
            "forearm_pronation": [80.0, 100.0],  # Pełny nachwyt
            "elbow_stability": [95.0, 100.0]
        },
        "trajectory_func": lambda t, lev, p="concentric": supported_wrist_curl_kinematics(t, lev, p, is_reverse=True),
        "act": {"forearms": act_forearm},
        "fibers": {"extensor_carpi": 1.00, "brachioradialis": 0.30, "flexor_carpi": 0.05}  # Czysta praca prostowników
    }


def evaluate_cable_reverse_wrist_curl(prof):
    levers = prof.get("levers", {})
    l_hand = 0.19

    weight = _get_weight(prof, "cable_reverse_wrist", 15.0)

    # Wyciąg daje ramię siły nawet na początku ruchu, naprawiając mankament sztangi
    total_m_arms = (l_hand * 0.5) * (0.6 + 0.4 * np.sin(t_vals * np.pi / 2))

    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.2, forearm_torque_share=1.00,
                                       forearm_activation=0.98, is_cable=True)
    act_forearm = min(0.98, raw_score / 18.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"cable_machine", "straight_bar"},
        "biomechanical_bounds": {
            "forearm_pronation": [80.0, 100.0],
            "torso_stability": [90.0, 100.0]
        },
        "trajectory_func": lambda t, lev, p="concentric": supported_wrist_curl_kinematics(t, lev, p, is_reverse=True),
        "act": {"forearms": act_forearm},
        "fibers": {"extensor_carpi": 1.00, "brachioradialis": 0.40}
    }


def evaluate_wrist_curl(prof):
    levers = prof.get("levers", {})
    l_hand = 0.19

    weight = _get_weight(prof, "wrist_curl", 25.0)
    weight_per_arm = weight / 2.0

    total_m_arms = (l_hand * 0.5) * np.sin(np.clip(t_vals + 0.2, 0, 1) * np.pi / 2)

    # Zginacze (podchwyt) są znacznie silniejsze niż prostowniki, dlatego waga idzie w górę, a próg do 25
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.0, forearm_torque_share=0.95,
                                       forearm_activation=0.95)
    act_forearm = min(0.95, raw_score / 25.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"barbell", "dumbbells", "bench"},
        "biomechanical_bounds": {
            "forearm_supination": [80.0, 100.0],  # Pełny podchwyt
            "elbow_stability": [95.0, 100.0]
        },
        "trajectory_func": lambda t, lev, p="concentric": supported_wrist_curl_kinematics(t, lev, p, is_reverse=False),
        "act": {"forearms": act_forearm},
        "fibers": {"flexor_carpi": 1.00, "palmaris_longus": 0.90, "extensor_carpi": 0.05}
        # Uderzenie w gruby brzusiec zginaczy
    }


def evaluate_behind_back_wrist_curl(prof):
    levers = prof.get("levers", {})
    l_hand = 0.19

    weight = _get_weight(prof, "behind_back_wrist", 40.0)  # Tu podnosi się więcej (lepsza pozycja)
    weight_per_arm = weight / 2.0

    total_m_arms = (l_hand * 0.5) * np.cos(t_vals * np.pi / 2.5)

    # Sztanga z tyłu lekko wyłącza palce, wymuszając pracę samego nadgarstka (flexor carpi ulnaris)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.5, forearm_torque_share=1.00,
                                       forearm_activation=0.98)
    act_forearm = min(1.00, raw_score / 35.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"barbell", "smith_machine"},
        "biomechanical_bounds": {
            "shoulder_extension_constant": [15.0, 30.0],  # Ręce z tyłu
            "elbow_extension_constant": [170.0, 180.0]
        },
        "trajectory_func": behind_back_wrist_curl_kinematics,
        "act": {"forearms": act_forearm},
        "fibers": {"flexor_carpi": 1.00, "palmaris_longus": 0.85}
    }


def evaluate_wrist_roller(prof):
    levers = prof.get("levers", {})

    weight = _get_weight(prof, "wrist_roller", 10.0)  # Zwykle wystarczy kilka kilogramów, żeby zniszczyć przedramiona
    weight_per_arm = weight / 2.0

    # Promień drążka jest malutki (np. 2 cm), ale praca jest ciągła i sumaryczna
    total_m_arms = np.full_like(t_vals, 0.02)

    # Wrist roller jest brutalny przez stałe napięcie i pracę naprzemienną
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.0, forearm_torque_share=1.00,
                                       forearm_activation=1.00, is_roller=True)
    # Zmniejszony dzielnik, bo mnożnik izometrii i stałego oporu robi swoje
    act_forearm = min(1.00, raw_score / 3.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"wrist_roller_device"},
        "biomechanical_bounds": {
            "shoulder_flexion_constant": [85.0, 95.0],  # Ręce uniesione w linii barków (praca przedniego aktonu!)
            "elbow_extension_constant": [165.0, 180.0]
        },
        "trajectory_func": wrist_roller_kinematics,
        "act": {"forearms": act_forearm, "delt_front": 0.80},  # Mocne izometryczne uderzenie w bark
        "fibers": {"extensor_carpi": 1.00, "flexor_carpi": 1.00, "delt_front": 0.90}
    }


def evaluate_hand_grippers(prof):
    levers = prof.get("levers", {})

    resistance = _get_weight(prof, "gripper_resistance", 45.0)  # Np. Captains of Crush (kg)

    total_m_arms = np.full_like(t_vals, 0.05)  # Ok. połowa dłoni dla palców naciskających rączkę

    # Izometria / krótki ROM miażdżący zginacze palców
    raw_score = calc_raw_physics_score(total_m_arms, resistance, rom_bonus=0.0, forearm_torque_share=1.00,
                                       forearm_activation=1.00, is_isometric=True)
    act_forearm = min(1.00, raw_score / 25.0)

    return {
        "cat": "Arms",
        "subcat": "isolation",
        "equipment": {"hand_grippers"},
        "biomechanical_bounds": {
            "wrist_neutral_stability": [95.0, 100.0]  # Nadgarstek nie może się łamać
        },
        "trajectory_func": gripper_kinematics,
        "act": {"forearms": act_forearm},
        "fibers": {"flexor_digitorum": 1.00, "flexor_carpi": 0.60}  # Dominują zginacze palców głębokie i powierzchowne
    }


def evaluate_pinch_grip_farmers_walk(prof):
    levers = prof.get("levers", {})
    body_weight = prof.get("weight_kg", 85.0)

    weight = _get_weight(prof, "pinch_grip", 15.0)  # Waga talerzy w dłoni

    total_m_arms = np.full_like(t_vals, 0.03)  # Opuszki palców (chwyt szczypcowy)

    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.0, forearm_torque_share=1.00,
                                       forearm_activation=1.00, is_isometric=True)
    act_forearm = min(1.00, raw_score / 5.0)

    return {
        "cat": "Arms",
        "subcat": "compound_isolation",
        "equipment": {"weight_plates"},
        "biomechanical_bounds": {
            "shoulder_depression": [95.0, 100.0],  # Kaptury w dół
            "thumb_engagement": [95.0, 100.0]  # Kluczowe użycie kciuka do szczypiec
        },
        "trajectory_func": pinch_grip_farmers_walk_kinematics,
        "act": {"forearms": act_forearm, "traps_upper": 0.60, "core": 0.70},
        "fibers": {"flexor_pollicis": 1.00, "flexor_digitorum": 0.90, "traps_upper": 0.75}
        # Zginacz długi kciuka odpala na 100%
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA PRZEDRAMIONA I CHWYT
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Reverse_Wrist_Curl": evaluate_reverse_wrist_curl(user_profile),
        "Cable_Reverse_Wrist_Curl": evaluate_cable_reverse_wrist_curl(user_profile),
        "Wrist_Curl": evaluate_wrist_curl(user_profile),
        "Behind_Back_Wrist_Curl": evaluate_behind_back_wrist_curl(user_profile),
        "Wrist_Roller": evaluate_wrist_roller(user_profile),
        "Hand_Grippers": evaluate_hand_grippers(user_profile),
        "Pinch_Grip_Farmers_Walk": evaluate_pinch_grip_farmers_walk(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na przedramiona i siłę chwytu. Wszystkie metryki kompletne.")
    except ImportError:
        pass