import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81


def _get_weight(prof, key, default_val):
    weights = prof.get("weights", {})
    return float(weights.get(key, default_val))


def calc_raw_physics_score(total_moment_arm, weight_per_leg_kg, rom_bonus,
                           quad_torque_share, quad_activation, penalty=1.0, is_isolation=False, is_sissy=False):
    """
    Krzywa oporu dla czworogłowych. Maksymalny moment siły występuje na dole (t=0) w rozciągnięciu.
    """
    knee_tau = (weight_per_leg_kg * G * total_moment_arm) * quad_torque_share

    # Maksymalny bonus za rozciągnięcie na dole ruchu (t=0)
    if is_sissy:
        stretch_bonus_factor = 1.0 + (rom_bonus * 2.0 * np.exp(-6 * t_vals))
    else:
        stretch_bonus_factor = 1.0 + (rom_bonus * 1.5 * np.exp(-10 * t_vals))

    # W izolacjach (Leg Extension) opór jest na całej długości (albo wręcz wyższy na górze)
    if is_isolation:
        leverage_factor = 0.5 + 0.5 * t_vals  # Ciężej na górze przy prostowaniu
    else:
        leverage_factor = 1.0 - 0.4 * t_vals  # Najciężej na dole w przysiadzie

    curve = knee_tau * stretch_bonus_factor * leverage_factor * quad_activation * penalty
    return trapz_func(curve, t_vals)


# =============================================================================
# 1. FUNKCJE KINEMATYKI I TRAJEKTORII (WZORCE BIOMETRYCZNE 3D)
# =============================================================================
# W ćwiczeniach na nogi `t=0` to najniższy punkt (maksymalne zgięcie/rozciągnięcie), 
# a `t=1` to pełne wyprostowanie. Ruch odbywa się głównie na osi Y (pionowo) 
# oraz Z (przód-tył biodra/kolana).

def back_squat_kinematics(t, levers, phase="concentric"):
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    # Przysiad: Y startuje nisko na dole (t=0), kończy na górze (t=1)
    y = -L_FEM * 0.90 * (1 - t)
    z = -L_FEM * 0.20 * (1 - t)  # Lekkie cofnięcie bioder
    return np.array([x, y, z])


def front_squat_kinematics(t, levers, phase="concentric"):
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    # Front squat pozwala na głębsze, bardziej pionowe zejście
    y = -L_FEM * 0.95 * (1 - t)
    z = L_FEM * 0.10 * (1 - t)  # Kolana idą mocniej do przodu, biodra mniej do tyłu
    return np.array([x, y, z])


def leg_press_kinematics(t, levers, phase="concentric"):
    """Wyciskanie na suwnicy pod kątem 45 stopni."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    # Tor ruchu zablokowany ukośnie
    y = -L_FEM * 0.70 * (1 - t)
    z = L_FEM * 0.70 * (1 - t)
    return np.array([x, y, z])


def hack_squat_kinematics(t, levers, phase="concentric"):
    """Maszyna Hack Squat: ruch po prowadnicy, plecy podparte."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    y = -L_FEM * 0.85 * (1 - t)
    z = L_FEM * 0.35 * (1 - t)
    return np.array([x, y, z])


def pendulum_squat_kinematics(t, levers, phase="concentric"):
    """Pendulum Squat: ruch po łuku wymuszający ogromną pracę czworogłowych."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    r = L_FEM * 1.5

    angle = np.radians(45) * (1 - t)
    x = HIP_W / 2
    y = -r * np.sin(angle)
    z = r * (1 - np.cos(angle))
    return np.array([x, y, z])


def bulgarian_split_squat_kinematics(t, levers, phase="concentric"):
    """Przysiad bułgarski: asymetryczny opad pionowy."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    y = -L_FEM * 0.90 * (1 - t)
    z = L_FEM * 0.15 * (1 - t)  # Noga przednia mocno zgięta w kolanie
    return np.array([x, y, z])


def leg_extension_kinematics(t, levers, phase="concentric"):
    """Prostowanie nóg siedząc: czysta rotacja w kolanie (izolacja)."""
    HIP_W = levers.get("hip_width", 0.32)
    L_TIB = levers.get("L_tibia", 0.38)

    x = HIP_W / 2
    # Start: kolano zgięte (t=0 -> ok. 90-100 stopni). Koniec: prosta noga (t=1).
    angle = np.radians(90) * (1 - t)
    y = -L_TIB * np.cos(angle)
    z = L_TIB * np.sin(angle)
    return np.array([x, y, z])


def lunges_kinematics(t, levers, phase="concentric"):
    """Wykroki/Zakroki."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    y = -L_FEM * 0.85 * (1 - t)
    z = L_FEM * 0.10 * (1 - t)
    return np.array([x, y, z])


def smith_machine_squat_kinematics(t, levers, phase="concentric"):
    """Przysiad na maszynie Smitha: rygorystycznie pionowy tor."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    y = -L_FEM * 0.85 * (1 - t)
    z = 0.0  # Całkowity brak ruchu w płaszczyźnie przód-tył dla sztangi
    return np.array([x, y, z])


def sissy_squat_kinematics(t, levers, phase="concentric"):
    """Przysiad Sissy: maksymalne wypchnięcie kolan w przód, odchylenie tułowia do tyłu."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)
    L_TIB = levers.get("L_tibia", 0.38)

    x = HIP_W / 2
    # Biodra idą w dół i mocno w przód
    y = -L_FEM * 0.95 * (1 - t)
    z = (L_FEM + L_TIB) * 0.50 * (1 - t)
    return np.array([x, y, z])


def belt_squat_kinematics(t, levers, phase="concentric"):
    """Belt Squat: obciążenie zwisa z pasa, pionowy ciąg bez obciążania kręgosłupa."""
    HIP_W = levers.get("hip_width", 0.32)
    L_FEM = levers.get("L_femur", 0.42)

    x = HIP_W / 2
    y = -L_FEM * 0.85 * (1 - t)
    z = -L_FEM * 0.10 * (1 - t)
    return np.array([x, y, z])


# =============================================================================
# 2. EWALUACJA OPOROWA I BIOMECHANICZNA DLA POSZCZEGÓLNYCH ĆWICZEŃ
# =============================================================================

def evaluate_back_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    # W przysiadzie podnosimy ciężar sztangi + około 88% wagi swojego ciała
    barbell_weight = _get_weight(prof, "back_squat", 100.0)
    weight_total = barbell_weight + (body_weight * 0.88)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_femur * np.cos(np.radians(35)) * (1 - 0.7 * t_vals)
    # W Back Squat pośladki i dwugłowe przejmują znaczną część momentu obrotowego bioder
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.1, quad_torque_share=0.55,
                                       quad_activation=0.90)
    act_quad = min(0.90, raw_score / 450.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell", "squat_rack"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [50.0, 80.0],
            "torso_forward_lean": [25.0, 45.0]
        },
        "trajectory_func": back_squat_kinematics,
        "act": {"quads": act_quad, "glutes": 0.85, "hamstrings": 0.40, "adductors": 0.50},
        "fibers": {"quads_rectus": 0.70, "quads_vastus": 0.90, "glute_maximus": 0.95}
    }


def evaluate_front_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    barbell_weight = _get_weight(prof, "front_squat", 80.0)
    weight_total = barbell_weight + (body_weight * 0.88)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_femur * np.cos(np.radians(25)) * (1 - 0.7 * t_vals)
    # Front Squat wymusza pionowy tułów, przenosząc load z pośladków na czworogłowe
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.3, quad_torque_share=0.75,
                                       quad_activation=0.98)
    act_quad = min(0.98, raw_score / 450.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"barbell", "squat_rack"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [40.0, 70.0],
            "torso_forward_lean": [5.0, 20.0]
        },
        "trajectory_func": front_squat_kinematics,
        "act": {"quads": act_quad, "glutes": 0.60, "core": 0.85},
        "fibers": {"quads_rectus": 0.85, "quads_vastus": 0.95, "glute_maximus": 0.70}
    }


def evaluate_leg_press(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)

    # Na suwnicy nie dźwigamy ciała, tylko sam wózek z ciężarem
    weight_total = _get_weight(prof, "leg_press", 200.0)
    weight_per_leg = weight_total / 2.0

    # Wektor siły jest zmniejszony o cos(45 st.) wózka
    effective_weight = weight_per_leg * np.cos(np.radians(45))
    total_m_arms = l_femur * 0.8 * (1 - 0.6 * t_vals)

    raw_score = calc_raw_physics_score(total_m_arms, effective_weight, rom_bonus=1.0, quad_torque_share=0.70,
                                       quad_activation=0.95)
    act_quad = min(0.95, raw_score / 500.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"leg_press_machine"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [60.0, 90.0],
            "hip_flexion_constant": [70.0, 100.0]
        },
        "trajectory_func": leg_press_kinematics,
        "act": {"quads": act_quad, "glutes": 0.60, "adductors": 0.40},
        "fibers": {"quads_rectus": 0.80, "quads_vastus": 0.95, "glute_maximus": 0.65}
    }


def evaluate_hack_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    machine_weight = _get_weight(prof, "hack_squat", 100.0)
    weight_total = machine_weight + (body_weight * 0.70)
    effective_weight = (weight_total / 2.0) * np.cos(np.radians(35))  # Kąt pochylenia maszyny

    total_m_arms = l_femur * 0.9 * (1 - 0.6 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, effective_weight, rom_bonus=1.4, quad_torque_share=0.85,
                                       quad_activation=1.00)
    act_quad = min(1.00, raw_score / 450.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"hack_squat_machine"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [40.0, 80.0],
            "torso_stability": [95.0, 100.0]
        },
        "trajectory_func": hack_squat_kinematics,
        "act": {"quads": act_quad, "glutes": 0.50},
        "fibers": {"quads_rectus": 0.90, "quads_vastus": 1.00, "glute_maximus": 0.60}
    }


def evaluate_pendulum_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    machine_weight = _get_weight(prof, "pendulum_squat", 60.0)
    weight_per_leg = (machine_weight + body_weight * 0.6) / 2.0

    # Łuk penduluma sprawia, że ramię momentu na kolanie jest stałe lub rośnie na dole
    total_m_arms = np.full_like(t_vals, l_femur * 1.0)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.8, quad_torque_share=0.95,
                                       quad_activation=1.00)
    act_quad = min(1.00, raw_score / 420.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"pendulum_squat_machine"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [30.0, 70.0],
            "ankle_dorsiflexion": [20.0, 45.0]
        },
        "trajectory_func": pendulum_squat_kinematics,
        "act": {"quads": act_quad, "glutes": 0.40},
        "fibers": {"quads_rectus": 0.95, "quads_vastus": 1.00, "glute_maximus": 0.50}
    }


def evaluate_bulgarian_split_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    dumbbells_weight = _get_weight(prof, "bulgarian_dumbbells", 40.0)
    # Cały ciężar spoczywa asymetrycznie w ok. 85% na nodze przedniej
    weight_front_leg = (dumbbells_weight + body_weight * 0.85)

    total_m_arms = l_femur * np.cos(np.radians(30)) * (1 - 0.7 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_front_leg, rom_bonus=1.3, quad_torque_share=0.65,
                                       quad_activation=0.95)
    act_quad = min(0.95, raw_score / 550.0)

    return {
        "cat": "Legs",
        "subcat": "unilateral",
        "equipment": {"dumbbells", "bench"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [50.0, 85.0],
            "torso_forward_lean": [15.0, 35.0]
        },
        "trajectory_func": bulgarian_split_squat_kinematics,
        "act": {"quads": act_quad, "glutes": 0.85, "core": 0.70},
        "fibers": {"quads_rectus": 0.85, "quads_vastus": 0.95, "glute_maximus": 0.90}
    }


def evaluate_leg_extension(prof):
    levers = prof.get("levers", {})
    l_tibia = levers.get("L_tibia", 0.38)

    machine_weight = _get_weight(prof, "leg_extension", 60.0)
    weight_per_leg = machine_weight / 2.0

    total_m_arms = l_tibia * np.cos(np.radians(20))
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=0.8, quad_torque_share=1.00,
                                       quad_activation=1.00, is_isolation=True)
    act_quad = min(1.00, raw_score / 280.0)

    return {
        "cat": "Legs",
        "subcat": "isolation",
        "equipment": {"leg_extension_machine"},
        "biomechanical_bounds": {
            "knee_extension_top": [170.0, 180.0],
            "hip_flexion_constant": [85.0, 95.0]
        },
        "trajectory_func": leg_extension_kinematics,
        "act": {"quads": act_quad},
        "fibers": {"quads_rectus": 1.00, "quads_vastus": 1.00}
    }


def evaluate_lunges(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    dumbbells_weight = _get_weight(prof, "lunges_dumbbells", 40.0)
    # Wykroki rozkładają ciężar nieco bardziej na obie nogi niż bułgary (ok. 75% na przednią)
    weight_front_leg = (dumbbells_weight + body_weight * 0.80) * 0.75

    total_m_arms = l_femur * np.cos(np.radians(20)) * (1 - 0.7 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_front_leg, rom_bonus=1.0, quad_torque_share=0.60,
                                       quad_activation=0.90)
    act_quad = min(0.90, raw_score / 480.0)

    return {
        "cat": "Legs",
        "subcat": "unilateral",
        "equipment": {"dumbbells"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [75.0, 95.0],
            "torso_straight_angle": [170.0, 180.0]
        },
        "trajectory_func": lunges_kinematics,
        "act": {"quads": act_quad, "glutes": 0.80, "hamstrings": 0.45},
        "fibers": {"quads_rectus": 0.80, "quads_vastus": 0.90, "glute_maximus": 0.85}
    }


def evaluate_smith_machine_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    smith_weight = _get_weight(prof, "smith_squat", 80.0)
    weight_total = smith_weight + (body_weight * 0.85)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_femur * 0.75 * (1 - 0.7 * t_vals)
    # Smith pozwala na wysunięcie stóp do przodu, eliminując pracę pośladków
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.1, quad_torque_share=0.80,
                                       quad_activation=0.95)
    act_quad = min(0.95, raw_score / 450.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"smith_machine"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [50.0, 85.0],
            "vertical_bar_path_deviation": [0.0, 2.0]
        },
        "trajectory_func": smith_machine_squat_kinematics,
        "act": {"quads": act_quad, "glutes": 0.60},
        "fibers": {"quads_rectus": 0.90, "quads_vastus": 0.95, "glute_maximus": 0.65}
    }


def evaluate_sissy_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    added_weight = _get_weight(prof, "sissy_added", 0.0)
    weight_per_leg = (added_weight + body_weight * 0.85) / 2.0

    # Największe ramię momentu dla kolana bez udziału bioder
    total_m_arms = l_femur * 1.5 * np.exp(-2 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=2.5, quad_torque_share=1.00,
                                       quad_activation=1.00, is_sissy=True)
    act_quad = min(1.00, raw_score / 420.0)

    return {
        "cat": "Legs",
        "subcat": "bodyweight_isolation",
        "equipment": {"sissy_squat_bench", "bodyweight"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [30.0, 60.0],
            "hip_extension_constant": [160.0, 180.0]  # Prosta linia tułów-udo
        },
        "trajectory_func": sissy_squat_kinematics,
        "act": {"quads": act_quad},
        "fibers": {"quads_rectus": 1.00, "quads_vastus": 1.00}
    }


def evaluate_belt_squat(prof):
    levers = prof.get("levers", {})
    l_femur = levers.get("L_femur", 0.42)
    body_weight = prof.get("weight_kg", 85.0)

    belt_weight = _get_weight(prof, "belt_squat", 120.0)
    # W Belt Squat odciążamy kręgosłup, waga ciała to tylko dolna połowa (ok. 60%)
    weight_total = belt_weight + (body_weight * 0.60)
    weight_per_leg = weight_total / 2.0

    total_m_arms = l_femur * np.cos(np.radians(25)) * (1 - 0.7 * t_vals)
    raw_score = calc_raw_physics_score(total_m_arms, weight_per_leg, rom_bonus=1.2, quad_torque_share=0.85,
                                       quad_activation=0.95)
    act_quad = min(0.95, raw_score / 460.0)

    return {
        "cat": "Legs",
        "subcat": "compound",
        "equipment": {"belt_squat_machine", "weight_belt"},
        "biomechanical_bounds": {
            "knee_flexion_bottom": [45.0, 75.0],
            "torso_forward_lean": [10.0, 30.0]
        },
        "trajectory_func": belt_squat_kinematics,
        "act": {"quads": act_quad, "glutes": 0.65},
        "fibers": {"quads_rectus": 0.85, "quads_vastus": 0.95, "glute_maximus": 0.70}
    }


# =============================================================================
# 3. GŁÓWNY SŁOWNIK BAZY DANYCH ĆWICZEŃ NA PRZÓD UDA
# =============================================================================

def get_exercises_data(user_profile):
    return {
        "Barbell_Back_Squat": evaluate_back_squat(user_profile),
        "Barbell_Front_Squat": evaluate_front_squat(user_profile),
        "Leg_Press": evaluate_leg_press(user_profile),
        "Hack_Squat": evaluate_hack_squat(user_profile),
        "Pendulum_Squat": evaluate_pendulum_squat(user_profile),
        "Bulgarian_Split_Squat": evaluate_bulgarian_split_squat(user_profile),
        "Leg_Extension": evaluate_leg_extension(user_profile),
        "Lunges": evaluate_lunges(user_profile),
        "Smith_Machine_Squat": evaluate_smith_machine_squat(user_profile),
        "Sissy_Squat": evaluate_sissy_squat(user_profile),
        "Belt_Squat": evaluate_belt_squat(user_profile)
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
            print(f"Załadowano {len(data)} ćwiczeń na mięśnie czworogłowe (przód uda). Wszystkie metryki kompletne.")
    except ImportError:
        pass