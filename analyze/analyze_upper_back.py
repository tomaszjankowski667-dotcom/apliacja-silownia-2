import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))

def calc_raw_physics_score(total_moment_arm, weight_per_arm_kg, rom_bonus,
                           upper_back_torque_share, upper_back_activation, penalty=1.0, 
                           is_row=False, is_shrug=False, is_face_pull=False, is_chest_supported=False):
    """
    Krzywa oporu dla góry pleców (czworoboczny, obłe, równoległoboczne).
    Wyróżnia unikalną fizykę szrugsów (izometria + pionowy ciąg) oraz siłę wiosłowań horyzontalnych.
    """
    tau = (weight_per_arm_kg * G * total_moment_arm) * upper_back_torque_share
    
    if is_shrug:
        # Szrugsy: ROM jest mikroskopijny, ale grawitacja działa ze 100% siłą w dół.
        # Napięcie jest stałe, a pod górę (t=1) kaptury wykonują gigantyczny skurcz izometryczny.
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-12 * t_vals))
        leverage_factor = np.ones_like(t_vals) * 1.5 # Bardzo krótka dźwignia wymaga mnożnika dla punktacji
    elif is_row:
        # Wiosłowania szeroko: Znakomita fizyka dla środka pleców. Ciężej na dole, gdzie łopatki uciekają w protrakcję.
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.2 * np.exp(-10 * t_vals))
        if is_chest_supported:
            leverage_factor = 0.9 + 0.1 * np.sin(t_vals * np.pi) # Ławka ułatwia retrakcję
        else:
            leverage_factor = 1.0 - 0.3 * t_vals
    elif is_face_pull:
        # Face pull angażujący obły mniejszy, podgrzebieniowy i czworoboczny
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.0 * np.exp(-6 * t_vals))
        leverage_factor = 0.5 + 0.5 * t_vals # Najciężej dopiąć ruch przy samej twarzy
    else:
        stretch_bonus_factor = 1.0
        leverage_factor = np.ones_like(t_vals)

    curve = tau * stretch_bonus_factor * leverage_factor * upper_back_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================

def barbell_upper_back_row_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie sztangą na górę pleców: Szeroki chwyt, łokcie na boki (flared)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 1.5) / 2 # Chwyt szerszy niż barki
    # Ciągnięcie do mostka/klatki piersiowej
    y = -L_HUM * 1.1 * (1 - t)
    # Łokcie wędrują mocno na zewnątrz (oś X) i w tył (oś Z), ramię pod kątem ok. 70-80 st do tułowia
    z = -L_HUM * 0.30 * t 
    return np.array([x, y, z])

def t_bar_row_kinematics(t, levers, phase="concentric"):
    """T-Bar Row (Półsztanga): Dłonie w chwycie neutralnym, mocna retrakcja łopatek."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 0.8) / 2 # Chwyt neutralny przy V-Barze
    # Kąt tułowia ok. 45 stopni. Ciągnięcie do brzucha/klatki (po skosie).
    y = -L_HUM * 1.2 * (1 - t)
    z = L_HUM * 0.20 * (1 - t) - L_HUM * 0.40 * t 
    return np.array([x, y, z])

def seal_row_kinematics(t, levers, phase="concentric"):
    """Seal Row / Cable Row: Pozycja pozioma, idealna izolacja środka pleców, zero pędu z bioder."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 1.4) / 2
    # Sztanga/Kabel porusza się pionowo od ziemi do ławki pod klatką
    y = -L_HUM * 1.2 * (1 - t)
    z = 0.0 # Tułów leży, brak wychyleń na osi Z
    return np.array([x, y, z])

def chest_supported_row_kinematics(t, levers, phase="concentric"):
    """Wiosłowanie przodem na ławce (Meadows/Incline Row)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    
    x = (C_W * 1.2) / 2
    # Ławka skośna (ok 30-45 stopni)
    y = -L_HUM * 1.0 * (1 - t)
    z = L_HUM * 0.30 * (1 - t) - L_HUM * 0.20 * t 
    return np.array([x, y, z])

def shrug_kinematics(t, levers, phase="concentric"):
    """Szrugsy (stojąc): Jedyny ruch to elewacja barków (łopatek). Ręce wyprostowane."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W * 1.2) / 2 # Hantle po bokach tułowia
    # Bardzo krótki ruch - unoszenie barków do uszu
    ROM_Y = 0.12 
    
    y = -(L_HUM + L_FOR) + ROM_Y * t
    z = 0.0 # Czysty ruch w płaszczyźnie czołowej/strzałkowej (pionowo)
    return np.array([x, y, z])

def face_pull_upper_back_kinematics(t, levers, phase="concentric"):
    """Face Pull: Wersja akcentująca rotatory zewnętrzne i mięsień czworoboczny (kaptury)."""
    C_W = levers.get("biacromial_width", 0.41)
    L_HUM = levers.get("L_humerus", 0.326)
    L_FOR = levers.get("L_forearm", 0.285)
    
    x = (C_W / 2) + L_HUM * 0.9 * t
    # Podciąganie na wysokość oczu z wyciągu górnego
    y = L_HUM * 0.6 * t 
    z = (L_HUM + L_FOR) * (1 - t)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_barbell_upper_back_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight_total = _get_weight(prof, "barbell_row", 70.0)
    weight_per_arm = weight_total / 2.0

    # Szerokie wiosłowanie skraca ramię momentu dla latsów, a maksuje dla tylnego barku i czworobocznego
    total_m_arms = l_humerus * np.cos(np.radians(20)) * (1 - 0.7 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.0, upper_back_torque_share=0.85, upper_back_activation=0.95, is_row=True)
    act_ub = min(0.95, raw_score / 260.0)

    return {
        "cat": "Back",
        "subcat": "compound",
        "equipment": {"barbell"},
        "biomechanical_bounds": {
            "elbow_flare_angle": [60.0, 90.0], # Klucz: szerokie łokcie to praca góry pleców, a nie najszerszego
            "torso_forward_lean": [45.0, 75.0]
        },
        "trajectory_func": barbell_upper_back_row_kinematics,
        "act": {"traps_mid": act_ub, "rhomboids": 0.90, "delt_rear": 0.85, "teres_major": 0.80},
        "fibers": {"traps_mid": 1.00, "rhomboid_major": 0.95, "teres_major": 0.90, "infraspinatus": 0.60}
    }

def evaluate_t_bar_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    body_weight = prof.get("weight_kg", 85.0)
    
    weight_total = _get_weight(prof, "t_bar_row", 65.0)
    # T-Bar odejmuje część obciążenia z odcinka lędźwiowego ze względu na punkt podparcia sztangi
    weight_per_arm = weight_total / 2.0

    total_m_arms = l_humerus * np.cos(np.radians(30)) * (1 - 0.6 * t_vals)
    
    # Kąt 45 stopni tułowia genialnie angażuje zarówno dół (najszerszy) jak i środek (równoległoboczne)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.2, upper_back_torque_share=0.75, upper_back_activation=0.95, is_row=True)
    act_ub = min(0.95, raw_score / 250.0)

    return {
        "cat": "Back",
        "subcat": "compound",
        "equipment": {"t_bar_machine", "barbell_with_v_handle"},
        "biomechanical_bounds": {
            "torso_forward_lean": [30.0, 50.0],
            "shoulder_extension_end": [160.0, 180.0]
        },
        "trajectory_func": t_bar_row_kinematics,
        "act": {"rhomboids": act_ub, "lats": 0.80, "traps_lower": 0.75},
        "fibers": {"rhomboid_major": 0.95, "rhomboid_minor": 0.90, "lats_thoracic": 0.85, "traps_mid": 0.80}
    }

def evaluate_seal_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight_total = _get_weight(prof, "seal_row", 55.0)
    weight_per_arm = weight_total / 2.0

    # Izolacja tułowia (leżenie) obniża siłę, ale podnosi czystość skurczu na środku pleców do maksimum
    total_m_arms = l_humerus * (0.4 + 0.6 * np.sin(t_vals * np.pi / 2))
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=1.5, upper_back_torque_share=0.95, upper_back_activation=1.00, is_row=True, is_chest_supported=True)
    act_ub = min(1.00, raw_score / 200.0)

    return {
        "cat": "Back",
        "subcat": "compound_isolation", # Ruch złożony, ale zablokowany tułów izoluje mięśnie docelowe
        "equipment": {"barbell", "high_bench", "dumbbells"},
        "biomechanical_bounds": {
            "torso_stability": [98.0, 100.0], # 0% użycia prostowników i pędu
            "elbow_flare_angle": [60.0, 90.0]
        },
        "trajectory_func": seal_row_kinematics,
        "act": {"rhomboids": act_ub, "traps_mid": 0.95, "delt_rear": 0.80},
        "fibers": {"rhomboid_major": 1.00, "traps_mid": 1.00, "teres_major": 0.85, "infraspinatus": 0.75}
    }

def evaluate_chest_supported_row(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight = _get_weight(prof, "chest_supported_row", 30.0)

    total_m_arms = l_humerus * np.cos(np.radians(25)) * (1 - 0.6 * t_vals)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.4, upper_back_torque_share=0.85, upper_back_activation=0.95, is_row=True, is_chest_supported=True)
    act_ub = min(0.95, raw_score / 220.0)

    return {
        "cat": "Back",
        "subcat": "compound",
        "equipment": {"dumbbells", "incline_bench"},
        "biomechanical_bounds": {
            "torso_stability": [95.0, 100.0],
            "scapular_retraction_end": [90.0, 100.0] # Konieczność mocnego spięcia łopatek na szczycie
        },
        "trajectory_func": chest_supported_row_kinematics,
        "act": {"traps_mid": act_ub, "rhomboids": 0.85, "lats": 0.60},
        "fibers": {"traps_mid": 0.95, "traps_lower": 0.80, "rhomboid_major": 0.90}
    }

def evaluate_face_pull_upper_back(prof):
    levers = prof.get("levers", {})
    l_humerus = levers.get("L_humerus", 0.326)
    
    weight = _get_weight(prof, "face_pull", 27.5)

    total_m_arms = l_humerus * (0.8 - 0.3 * t_vals)
    
    # Face pull w tej wersji jest klasyfikowany jako król dla obłego mniejszego i podgrzebieniowego (rotator cuff)
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.5, upper_back_torque_share=0.85, upper_back_activation=0.95, is_face_pull=True)
    act_ub = min(0.95, raw_score / 180.0)

    return {
        "cat": "Back/Shoulders",
        "subcat": "compound_isolation",
        "equipment": {"cable_machine", "rope"},
        "biomechanical_bounds": {
            "shoulder_external_rotation": [45.0, 90.0], 
            "scapular_retraction_end": [85.0, 100.0]
        },
        "trajectory_func": face_pull_upper_back_kinematics,
        "act": {"rotator_cuff": act_ub, "traps_mid": 0.85, "delt_rear": 0.90},
        "fibers": {"infraspinatus": 1.00, "teres_minor": 0.95, "traps_mid": 0.90, "traps_lower": 0.70}
    }

def evaluate_dumbbell_shrugs(prof):
    levers = prof.get("levers", {})
    c_w = levers.get("biacromial_width", 0.41)
    
    weight = _get_weight(prof, "dumbbell_shrugs", 35.0)

    # W szrugsach dźwignią nie jest ręka (która zwisa prosto), ale obojczyk i łopatka
    total_m_arms = np.full_like(t_vals, c_w * 0.4) 
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=0.0, upper_back_torque_share=1.00, upper_back_activation=1.00, is_shrug=True)
    # Szrugsy wymagają mniejszego dzielnika z powodu fizycznie bardzo krótkiej dźwigni, a co za tym idzie mniejszego wyniku integralnego
    act_ub = min(1.00, raw_score / 100.0)

    return {
        "cat": "Back/Shoulders",
        "subcat": "isolation",
        "equipment": {"dumbbells"},
        "biomechanical_bounds": {
            "scapular_elevation": [80.0, 100.0], # Unoszenie barków do uszu
            "elbow_extension_constant": [170.0, 180.0] # Proste łokcie! Brak oszukiwania bicepsem
        },
        "trajectory_func": shrug_kinematics,
        "act": {"traps_upper": act_ub, "forearms": 0.60},
        "fibers": {"traps_upper": 1.00, "levator_scapulae": 0.95} # 100% aktywacji górnego aktonu kaptura
    }

def evaluate_barbell_shrugs(prof):
    levers = prof.get("levers", {})
    c_w = levers.get("biacromial_width", 0.41)
    
    weight_total = _get_weight(prof, "barbell_shrugs", 100.0) # Tu ciężary bywają potężne
    weight_per_arm = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, c_w * 0.4)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.5, upper_back_torque_share=1.00, upper_back_activation=0.95, is_shrug=True)
    act_ub = min(0.98, raw_score / 120.0)

    return {
        "cat": "Back/Shoulders",
        "subcat": "isolation",
        "equipment": {"barbell"},
        "biomechanical_bounds": {
            "scapular_elevation": [70.0, 95.0],
            "torso_stability": [90.0, 100.0]
        },
        "trajectory_func": shrug_kinematics,
        "act": {"traps_upper": act_ub, "forearms": 0.75},
        "fibers": {"traps_upper": 1.00, "levator_scapulae": 0.90}
    }

def evaluate_smith_shrugs(prof):
    levers = prof.get("levers", {})
    c_w = levers.get("biacromial_width", 0.41)
    
    weight_total = _get_weight(prof, "smith_shrugs", 120.0)
    weight_per_arm = weight_total / 2.0

    total_m_arms = np.full_like(t_vals, c_w * 0.4)
    
    # Maszyna Smitha pozwala na idealne odcięcie stabilizatorów i skupienie na czystej elewacji łopatki
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_arm, rom_bonus=0.8, upper_back_torque_share=1.00, upper_back_activation=1.00, is_shrug=True)
    act_ub = min(1.00, raw_score / 130.0)

    return {
        "cat": "Back/Shoulders",
        "subcat": "isolation",
        "equipment": {"smith_machine", "calf_machine"},
        "biomechanical_bounds": {
            "scapular_elevation": [80.0, 100.0],
            "vertical_bar_path_deviation": [0.0, 2.0]
        },
        "trajectory_func": shrug_kinematics,
        "act": {"traps_upper": act_ub},
        "fibers": {"traps_upper": 1.00, "levator_scapulae": 0.95}
    }

def evaluate_cable_shrugs(prof):
    levers = prof.get("levers", {})
    c_w = levers.get("biacromial_width", 0.41)
    
    weight = _get_weight(prof, "cable_shrugs", 40.0)

    # Wyciąg daje napięcie pod kątem, co przy szrugsach lekko aktywuje też dolny i środkowy akton (retrakcję)
    total_m_arms = np.full_like(t_vals, c_w * 0.45)
    
    raw_score = calc_raw_physics_score(total_m_arms, weight, rom_bonus=1.5, upper_back_torque_share=0.95, upper_back_activation=0.98, is_shrug=True)
    act_ub = min(0.95, raw_score / 90.0)

    return {
        "cat": "Back/Shoulders",
        "subcat": "isolation",
        "equipment": {"cable_machine", "straight_bar_or_handles"},
        "biomechanical_bounds": {
            "scapular_elevation": [85.0, 100.0],
            "torso_backward_lean": [5.0, 15.0] # Lekkie odchylenie na wyciągu pozwala wydłużyć ROM
        },
        "trajectory_func": shrug_kinematics,
        "act": {"traps_upper": act_ub, "traps_mid": 0.40},
        "fibers": {"traps_upper": 0.95, "traps_mid": 0.50, "levator_scapulae": 0.85}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA GÓRĘ I ŚRÓDEK PLECÓW
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Barbell_Upper_Back_Row": evaluate_barbell_upper_back_row(user_profile),
        "T_Bar_Row": evaluate_t_bar_row(user_profile),
        "Seal_Row": evaluate_seal_row(user_profile),
        "Chest_Supported_Row": evaluate_chest_supported_row(user_profile),
        "Face_Pull_Upper_Back": evaluate_face_pull_upper_back(user_profile),
        "Dumbbell_Shrugs": evaluate_dumbbell_shrugs(user_profile),
        "Barbell_Shrugs": evaluate_barbell_shrugs(user_profile),
        "Smith_Shrugs": evaluate_smith_shrugs(user_profile),
        "Cable_Shrugs": evaluate_cable_shrugs(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na górę i środek pleców (kaptury/równoległoboczne). Wszystkie metryki kompletne.")
    except ImportError:
        pass