"""
ERECTORS ANALYZER & EXPORTER (analyze_erectors.py)
--------------------------------------------------
Oblicza Hypertrophy Score dla 5 ćwiczeń na PROSTOWNIKI KRĘGOSŁUPA.
Oddziela napięcie izometryczne od dynamicznego i surowo ocenia
koszty neurologiczne (CNS fatigue) dla każdego boju.
Wyniki zapisuje do pliku tekstowego 'erectors_scores.txt'.
"""

import numpy as np

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
t_vals = np.linspace(0, 1, 100)
G = 9.81

# --- BAZA PROFILI DLA PROSTOWNIKÓW ---
# L_torso: Długość tułowia (kluczowa dźwignia dla dolnego odcinka pleców)
ERECTOR_PROFILES = {
    "Twój Profil (Długi Tułów, Długie Nogi)": {
        "L_torso": 0.48, "weight_bw": 85,
        "weights": {"deadlift": 140, "rack_pull": 170, "back_ext": 20, "good_morning": 60, "squat": 100}
    },
    "T-Rex (Krótki, potężny tułów)": {
        "L_torso": 0.42, "weight_bw": 95,
        "weights": {"deadlift": 180, "rack_pull": 220, "back_ext": 35, "good_morning": 90, "squat": 160}
    },
    "Average Joe (Średni)": {
        "L_torso": 0.45, "weight_bw": 80,
        "weights": {"deadlift": 130, "rack_pull": 150, "back_ext": 15, "good_morning": 50, "squat": 120}
    }
}


# --- SILNIK OBLICZENIOWY DLA PROSTOWNIKÓW ---
def get_weight(prof, key, default=50):
    return prof.get("weights", {}).get(key, default)


def calc_erectors_stimulus(moment_arm, weight_kg, torque_share, rom_stretch_bonus,
                           cns_fatigue_penalty, internal_curve, is_isometric=True, dynamic_bonus=1.0):
    force_newtons = weight_kg * G
    total_tau = force_newtons * moment_arm

    # Jaką część pracy wykonują prostowniki względem pośladków/dwugłowych
    erector_tau = total_tau * torque_share

    # Praca dynamiczna (zgięcie i wyprost kręgosłupa) daje znacznie wyższy bodziec do wzrostu niż izometria
    if not is_isometric:
        stretch_factor = 1.0 + (rom_stretch_bonus * 1.8 * np.exp(-10 * t_vals)) * dynamic_bonus
    else:
        # Przy izometrii mięsień nie ulega rozciągnięciu w trakcie ruchu pod obciążeniem (brak uszkodzeń Z-line)
        stretch_factor = 1.0 + (rom_stretch_bonus * 0.4 * np.exp(-10 * t_vals))

    # Układ nerwowy to główne wąskie gardło dla dołu pleców (CNS fatigue)
    neural_drive = cns_fatigue_penalty

    return erector_tau * internal_curve * stretch_factor * neural_drive


# --- DEFINICJE 5 WZORCÓW RUCHOWYCH ---

def get_conventional_deadlift(prof):
    # Martwy Ciąg Klasyczny.
    # Gigantyczne ramię siły na dole (tułów pochylony). Ogromny ciężar absolutny.
    m_arm = prof["L_torso"] * np.cos(np.radians(20)) * (1 - 0.7 * t_vals)
    weight = get_weight(prof, "deadlift", 140)
    torque_share = 0.50  # Dzieli pracę z potężnymi pośladkami i tyłem uda

    internal_curve = np.exp(-((t_vals - 0.1) / 0.4) ** 2)
    # KARA CNS: Zbyt wyczerpujące, by używać jako podstawy objętości hipertroficznej
    return calc_erectors_stimulus(m_arm, weight, torque_share, rom_stretch_bonus=1.2,
                                  cns_fatigue_penalty=0.60, internal_curve=internal_curve, is_isometric=True)


def get_rack_pull(prof):
    # Martwy ciąg z podwyższenia (nad kolanem).
    # Ramię siły mniejsze (tułów bardziej pionowo), ale ciężar astronomiczny.
    m_arm = prof["L_torso"] * np.cos(np.radians(45)) * (1 - 0.5 * t_vals)
    weight = get_weight(prof, "rack_pull", 170)
    torque_share = 0.60  # Mniejszy udział nóg, większy nacisk na górę pleców i prostowniki

    internal_curve = np.exp(-1.0 * t_vals)
    return calc_erectors_stimulus(m_arm, weight, torque_share, rom_stretch_bonus=0.5,
                                  cns_fatigue_penalty=0.65, internal_curve=internal_curve, is_isometric=True)


def get_back_extension_45(prof):
    # Wyprosty tułowia na ławce rzymskiej (45 stopni).
    # Można wykonywać z intencją zgięcia kręgosłupa (dynamicznie).
    m_arm = prof["L_torso"] * np.cos((t_vals - 0.2) * (np.pi / 2))
    # Ciężar to masa górnej połowy ciała (~55% masy) + obciążenie zewnętrzne
    weight = (prof["weight_bw"] * 0.55) + get_weight(prof, "back_ext", 20)
    torque_share = 0.75  # Przy zaokrąglaniu pleców prostowniki przejmują ogromną część pracy od pośladków

    internal_curve = np.full_like(t_vals, 1.0)
    # BRAK KARY CNS: Można robić w seriach po 15-20 powtórzeń do całkowitego załamania
    return calc_erectors_stimulus(m_arm, weight, torque_share, rom_stretch_bonus=1.5,
                                  cns_fatigue_penalty=1.10, internal_curve=internal_curve, is_isometric=False,
                                  dynamic_bonus=1.5)


def get_good_morning(prof):
    # Dzień dobry ze sztangą.
    # Wektor siły ekstremalnie oddalony od bioder (sztanga na karku). Najdłuższe możliwe ramię siły.
    m_arm = prof["L_torso"] * 1.2 * (1 - 0.6 * t_vals)
    weight = get_weight(prof, "good_morning", 60)
    torque_share = 0.55

    internal_curve = np.exp(-1.5 * t_vals)
    # Zazwyczaj wykonywane izometrycznie dla kręgosłupa (ruch z biodra)
    return calc_erectors_stimulus(m_arm, weight, torque_share, rom_stretch_bonus=1.3,
                                  cns_fatigue_penalty=0.80, internal_curve=internal_curve, is_isometric=True)


def get_back_squat(prof):
    # Przysiad ze sztangą.
    # Tułów jest relatywnie wyprostowany (kąt 60-75 stopni), więc ramię siły na dół pleców jest umiarkowane.
    m_arm = prof["L_torso"] * np.cos(np.radians(60)) * (1 - 0.5 * t_vals)
    weight = get_weight(prof, "squat", 100)
    torque_share = 0.30  # Nogi i pośladki wykonują 70% pracy

    internal_curve = np.exp(-((t_vals - 0.2) / 0.4) ** 2)
    return calc_erectors_stimulus(m_arm, weight, torque_share, rom_stretch_bonus=0.8,
                                  cns_fatigue_penalty=0.70, internal_curve=internal_curve, is_isometric=True)


EXERCISES = [
    ("Martwy Ciąg Klasyczny", get_conventional_deadlift),
    ("Martwy Ciąg z Podwyższenia (Rack Pull)", get_rack_pull),
    ("Wyprosty Tułowia (Ławka Rzymska 45°)", get_back_extension_45),
    ("Dzień Dobry (Good Morning)", get_good_morning),
    ("Przysiad ze Sztangą (Izometria)", get_back_squat)
]


# --- ZAPIS DO PLIKU ---
def export_scores_to_file():
    global_max_score = 0

    for _, ex_func in EXERCISES:
        for _, prof_data in ERECTOR_PROFILES.items():
            score = trapz_func(ex_func(prof_data), t_vals)
            if score > global_max_score:
                global_max_score = score

    THRESHOLD_SCORE = global_max_score * 0.15

    with open("erectors_scores.txt", "w", encoding="utf-8") as f:
        f.write("Baza Danych: Hypertrophy Score (PROSTOWNIKI KRĘGOSŁUPA)\n")
        f.write("=" * 65 + "\n")

        for prof_name, prof_data in ERECTOR_PROFILES.items():
            f.write(f"\n[USER]: {prof_name}\n")
            f.write("-" * 40 + "\n")

            for ex_name, ex_func in EXERCISES:
                raw_score = trapz_func(ex_func(prof_data), t_vals)

                if raw_score < THRESHOLD_SCORE:
                    f.write(f"{ex_name}: ODRZUT (Score: 0.0)\n")
                else:
                    display_score = min(10.0, max(0.5, (raw_score / global_max_score) * 10))
                    f.write(f"{ex_name}: {display_score:.1f}\n")


if __name__ == "__main__":
    export_scores_to_file()
    print("Zapisano wyniki do 'erectors_scores.txt'.")