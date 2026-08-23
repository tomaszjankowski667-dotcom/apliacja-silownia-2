import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_kg, rom_bonus,
                           delt_torque_share, delt_activation, penalty=1.0, 
                           is_fw_raise=False, is_cable=False, is_machine=False, 
                           is_face_pull=False, is_row=False):
    """
    Krzywa oporu dla tylnego aktonu barku.
    Główna różnica polega na przebiegu napięcia: hantle mają 0 na dole, a maszyny/kable pełne obciążenie.
    """
    tau = (weight_kg * G * total_moment_arm) * delt_torque_share
    
    if is_fw_raise:
        # Hantle w opadzie tułowia: zero na dole (zwis swobodny), maksimum na górze (skurcz)
        stretch_bonus_factor = 1.0 
        leverage_factor = np.sin(t_vals * np.pi / 2) # Wzrasta od 0 do 1
    elif is_machine:
        # Reverse Pec-Deck: Stałe, idealne napięcie przez cały zakres
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-8 * t_vals)) # Ekstremalnie dobre rozciągnięcie w fazie początkowej
        leverage_factor = np.ones_like(t_vals)
    elif is_cable:
        # Cross-Cable (linki na krzyż): Linki krzyżują się przed ciałem, świetne napięcie startowe
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.8 * np.exp(-6 * t_vals))
        leverage_factor = 0.8 + 0.2 * np.sin(t_vals * np.pi)
    elif is_face_pull:
        # Face Pull: Linka z przodu-góry ciągnięta do twarzy. Silny skurcz izometryczny i rotacja zewnętrzna na górze (t=1).
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.0 * np.exp(-6 * t_vals))
        leverage_factor = 0.5 + 0.5 * t_vals # Najciężej dociągnąć do samej twarzy
    elif is_row:
        # Wiosłowanie szeroko: Wielostawowe, włącza całe plecy. Ciężej na dole, lżej dociągnąć na maksa
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-10 * t_vals))
        leverage_factor = 1.0 - 0.3 * t_vals
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * delt_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def reverse_pec_deck_kinematics(t, levers, phase="concentric"):
    """Odwrotne rozpiętki na maszynie (Butterfly): Ruch horyzontalny po stałym łuku."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.95 # Lekko ugięte łokcie
    
    # Kąt od ok. 90 stopni (ręce w przód) do 0 stopni (ręce w bok wzdłuż linii barków)
    angle = np.radians(90) * (1 - t)
    x = (C_W / 2) + r * np.cos(angle)
    y = 0.0 # Siedząc, brak ruchu pionowego (płaszczyzna pozioma)
    z = r * np.sin(angle) # Ruch od przodu (Z duże) na boki (Z dążące do 0)
    return np.array([x, y, z])

def reverse_cable_flyes_kinematics(t, levers, phase="concentric"):
    """Odwrotne rozpiętki na wyciągu (Cross-Cable): Kable krzyżują się przed ciałem."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = L_HUM + L_FOR
    
    # Ręce krzyżują się na starcie (t=0, kąt w okolicach 110 stopni od linii barków)
    angle = np.radians(110) * (1 - t)
    x = (C_W / 2) + r * np.cos(angle)
    y = -0.10 * t # Lekki ruch w dół w zależności od ustawienia wyciągu
    z = r * np.sin(angle)
    return np.array([x, y, z])

def bent_over_lateral_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy hantli w opadzie tułowia: Opad o ok. 90 stopni, walka z grawitacją z dołu do boku."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.90
    
    x = (C_W / 2) + r * np.sin(np.radians(90) * t)
    # Ze zwisu (t=0) w górę (t=1). Tułów jest poziomo, więc ruch względem ziemi jest pionowy.
    y = -r * np.cos(np.radians(90) * t) 
    z = 0.10 * t # Lekko w kierunku głowy ze względu na angaż bocznego/tylnego aktonu
    return np.array([x, y, z])

def incline_bench_rear_delt_raise_kinematics(t, levers, phase="concentric"):
    """Wznosy hantli w leżeniu przodem na ławce skośnej: Stabilizacja tułowia zdejmuje pęd (cheat)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    r = (L_HUM + L_FOR) * 0.95
    
    # Tułów na skosie ok. 30-45 stopni. Kinematyka podobna do opadu, ale surowsza rzutowo.
    angle = np.radians(90) * t
    x = (C_W / 2) + r * np.sin(angle)
    y = -r * np.cos(angle)
    z = 0.05 * t
    return np.array([x, y, z])

def face_pull_kinematics(t, levers, phase="concentric"):
    """Face Pull (Wyciąg do twarzy): Kombinacja zgięcia horyzontalnego i skrajnej rotacji zewnętrznej."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    # Z rąk wyprostowanych do łokci szeroko z boku, a dłoni na wysokości oczu (rotacja)
    x = (C_W / 2) + L_HUM * 0.8 * t
    # Podciąganie na wysokość czoła/oczu
    y = L_HUM * 0.5 * t 
    # Linka jest przyciągana od przodu (t=0) do samej twarzy (t=1)
    z = (L_HUM + L_FOR) * (1 - t)
    return np.array([x, y, z])

def wide_grip_row_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie w opadzie tułowia, szeroki chwyt: Łokcie pod kątem 90 stopni do ciała."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 1.6) / 2 # Chwyt szeroki
    # Przyciąganie hantli/sztangi z dołu pionowo do klatki piersiowej
    y = -L_HUM * 1.2 * (1 - t)
    # Łokcie wędrują mocno na zewnątrz i do tyłu
    z = -L_HUM * 0.20 * t 
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_reverse_pec_deck(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    machine_weight = _get_weight(prof, "reverse_pec_deck", 45.0)
    weight_per_arm = machine_weight / 2.0

    # Dźwignia to wyciągnięte ręce, ramię momentu pozostaje stałe dzięki krzywce maszyny
    total_m_arms = np.full_like(t_vals, (l_humerus + l_forearm) * 0.85)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=2.0, delt_torque_share=0.95, delt_activation=1.00, is_machine=True)
    act_rear = min(1.00, raw_score / 220.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"pec_deck_machine"},
        "biomechanical_bounds": {
            "shoulder_horizontal_abduction": [0.0, 90.0], # Od przodu do boku
            "elbow_flexion_constant": [160.0, 180.0] # Prawie proste ręce
        },
        "trajectory_func": reverse_pec_deck_kinematics,
        "act": {"delt_rear": act_rear, "traps_mid": 0.60, "rhomboids": 0.65},
        "fibers": {"delt_rear": 1.00, "traps_mid": 0.70, "infraspinatus": 0.50}
    }

def evaluate_reverse_cable_flyes(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    cable_weight = _get_weight(prof, "reverse_cable_flyes", 15.0)

    total_m_arms = (l_humerus + l_forearm) * (0.8 + 0.2 * np.sin(t_vals * np.pi))
    
    # Linki z krzyża zapewniają fantastyczne rozciągnięcie przyśrodkowe i opór od samej fazy startowej
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=2.5, delt_torque_share=0.98, delt_activation=1.00, is_cable=True)
    act_rear = min(1.00, raw_score / 180.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"cable_machine"},
        "biomechanical_bounds": {
            "shoulder_horizontal_abduction": [-20.0, 90.0], # Zaczyna ze skrzyżowanymi rękami
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": reverse_cable_flyes_kinematics,
        "act": {"delt_rear": act_rear, "delt_lateral": 0.35, "rhomboids": 0.60},
        "fibers": {"delt_rear": 1.00, "delt_lateral": 0.40, "infraspinatus": 0.60}
    }

def evaluate_bent_over_lateral_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    dumbbell_weight = _get_weight(prof, "db_rear_delt_raise", 12.5)

    # Ramię momentu rośnie od zera na dole do maksa w poziomie (sin 0 -> sin 90)
    total_m_arms = (l_humerus + l_forearm * 0.9) * np.sin(t_vals * np.pi / 2)
    
    raw_score = calc_raw_physics_score(total_m_arms, dumbbell_weight, rom_bonus=0.0, delt_torque_share=0.85, delt_activation=0.90, is_fw_raise=True)
    act_rear = min(0.92, raw_score / 150.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"dumbbells"},
        "biomechanical_bounds": {
            "torso_forward_lean": [70.0, 90.0], # Głęboki opad tułowia
            "shoulder_horizontal_abduction": [0.0, 90.0]
        },
        "trajectory_func": bent_over_lateral_raise_kinematics,
        "act": {"delt_rear": act_rear, "traps_mid": 0.70, "delt_lateral": 0.40},
        "fibers": {"delt_rear": 1.00, "traps_mid": 0.85, "rhomboid_major": 0.70}
    }

def evaluate_incline_bench_rear_delt_raise(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    l_forearm = levers.get("L_forearm", 0.285)
    
    dumbbell_weight = _get_weight(prof, "incline_rear_delt_raise", 10.0)

    total_m_arms = (l_humerus + l_forearm * 0.9) * np.sin(t_vals * np.pi / 2)
    
    # Ławka eliminuje bujanie ciałem, gwarantując 100% obciążenia na mięsień docelowy. 
    # Napięcie aktywacyjne idzie w górę dzięki braku oszukiwania (cheatingu).
    raw_score = calc_raw_physics_score(total_m_arms, dumbbell_weight, rom_bonus=0.0, delt_torque_share=0.95, delt_activation=0.98, is_fw_raise=True)
    act_rear = min(0.98, raw_score / 130.0)

    return {
        "cat": "Shoulders",
        "subcat": "isolation",
        "equipment": {"dumbbells", "incline_bench"},
        "biomechanical_bounds": {
            "torso_stability": [98.0, 100.0], # Pełna izolacja przez leżenie
            "shoulder_horizontal_abduction": [0.0, 90.0]
        },
        "trajectory_func": incline_bench_rear_delt_raise_kinematics,
        "act": {"delt_rear": act_rear, "rhomboids": 0.65},
        "fibers": {"delt_rear": 1.00, "rhomboid_minor": 0.75, "infraspinatus": 0.55}
    }

def evaluate_face_pull(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    cable_weight = _get_weight(prof, "face_pull", 27.5)

    # Dźwignia skrócona na przedramieniu z powodu zgięcia łokciowego, ale 
    # potężny moment obrotowy na zewnętrznych rotatorach (łokcie odciągane w bok)
    total_m_arms = l_humerus * (0.8 - 0.3 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, cable_weight, rom_bonus=0.5, delt_torque_share=0.75, delt_activation=0.95, is_face_pull=True)
    act_rear = min(0.95, raw_score / 180.0)

    return {
        "cat": "Shoulders",
        "subcat": "compound_isolation", # Izoluje bark, ale wymaga pracy pleców i bicepsa
        "equipment": {"cable_machine", "rope"},
        "biomechanical_bounds": {
            "shoulder_external_rotation": [45.0, 90.0], # Kluczowy element Face Pulla - rotacja nadgarstków w górę
            "elbow_flare_angle": [85.0, 100.0] # Łokcie szeroko
        },
        "trajectory_func": face_pull_kinematics,
        "act": {"delt_rear": act_rear, "traps_mid": 0.85, "rotator_cuff": 0.95},
        "fibers": {"delt_rear": 1.00, "infraspinatus": 1.00, "teres_minor": 0.95, "traps_mid": 0.90}
    }

def evaluate_wide_grip_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    barbell_weight = _get_weight(prof, "wide_grip_row", 60.0)
    weight_per_arm = barbell_weight / 2.0

    # Najkrótsza dźwignia boczna, ale gigantyczny ciężar użyty w ćwiczeniu
    total_m_arms = l_humerus * np.cos(np.radians(20)) * (1 - 0.7 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.8, delt_torque_share=0.45, delt_activation=0.85, is_row=True)
    act_rear = min(0.90, raw_score / 250.0)

    return {
        "cat": "Back/Shoulders",
        "subcat": "compound",
        "equipment": {"barbell", "dumbbells"},
        "biomechanical_bounds": {
            "elbow_flare_angle": [70.0, 90.0], # Szeroki chwyt to klucz do aktywacji tyłu barku zamiast najszerszego
            "torso_forward_lean": [45.0, 80.0]
        },
        "trajectory_func": wide_grip_row_kinematics,
        "act": {"delt_rear": act_rear, "traps_mid": 0.95, "rhomboids": 0.90, "lats": 0.40},
        "fibers": {"delt_rear": 0.85, "traps_mid": 1.00, "rhomboid_major": 0.95, "lats_upper": 0.50}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA TYŁ BARKU
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Reverse_Pec_Deck": evaluate_reverse_pec_deck(user_profile),
        "Reverse_Cable_Flyes": evaluate_reverse_cable_flyes(user_profile),
        "Bent_Over_Lateral_Raise": evaluate_bent_over_lateral_raise(user_profile),
        "Incline_Rear_Delt_Raise": evaluate_incline_bench_rear_delt_raise(user_profile),
        "Face_Pull": evaluate_face_pull(user_profile),
        "Wide_Grip_Row": evaluate_wide_grip_row(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na tylny akton barków. Wszystkie metryki kompletne.")
    except ImportError:
        pass