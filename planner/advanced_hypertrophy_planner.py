"""
ADVANCED HYPERTROPHY PLANNER (McDonald's Yield Model & Fractional Fatigue)
--------------------------------------------------------------------------
1. Optymalizuje trening pod kątem MAKSYMALNEGO PRZYROSTU MASY (w gramach/tydzień).
2. Nieużywane historycznie włókna otrzymują mnożnik wzrostu nowicjusza (x12).
3. Seria ćwiczenia z aktywacją < 0.5 kosztuje ułamek budżetu zmęczenia (Fatigue Discount).
4. Brzuch i łydki są izolowane tylko dla osób o stażu > 1.5 roku.
"""

import numpy as np
from scipy.optimize import linprog
import json


class AdvancedHypertrophyPlanner:
    def __init__(self, years_of_training, recovery_factor):
        self.stage = years_of_training
        self.recovery = recovery_factor

        # Limity czasowe i treningowe
        self.TIME_PER_SET_MIN = 3.5
        self.MAX_SETS_SESSION = 22

        # 1. BAZA AKTONÓW: MRV oraz HISTORYCZNE ZAANGAŻOWANIE (0.0 - 1.0)
        # historical_engagement = 0.9 oznacza, że akton był "katowany" przez lata.
        # historical_engagement = 0.1 oznacza, że włókna są "dziewicze" (np. dół klatki).
        self.AKTONS = {
            "chest_upper": {"mrv": 12.0, "historical_engagement": 0.30},
            "chest_mid": {"mrv": 14.0, "historical_engagement": 0.95},
            "chest_lower": {"mrv": 10.0, "historical_engagement": 0.15},
            "delt_front": {"mrv": 8.0, "historical_engagement": 0.95},
            "delt_side": {"mrv": 16.0, "historical_engagement": 0.40},
            "delt_rear": {"mrv": 14.0, "historical_engagement": 0.20},
            "tricep_long": {"mrv": 10.0, "historical_engagement": 0.30},
            "tricep_lateral": {"mrv": 12.0, "historical_engagement": 0.85},
            "bicep_long": {"mrv": 10.0, "historical_engagement": 0.80},
            "bicep_short": {"mrv": 12.0, "historical_engagement": 0.50},
            "abs_upper": {"mrv": 10.0, "historical_engagement": 0.80},
            "abs_lower": {"mrv": 10.0, "historical_engagement": 0.20},
            "abs_oblique": {"mrv": 8.0, "historical_engagement": 0.10},
            "quad_vastus": {"mrv": 14.0, "historical_engagement": 0.90},
            "quad_rectus": {"mrv": 12.0, "historical_engagement": 0.50},
            "hamstring": {"mrv": 12.0, "historical_engagement": 0.40},
            "calves": {"mrv": 14.0, "historical_engagement": 0.10},
            "back_lats": {"mrv": 14.0, "historical_engagement": 0.85},
            "back_mid": {"mrv": 14.0, "historical_engagement": 0.40},
            "back_lower": {"mrv": 10.0, "historical_engagement": 0.60}
        }

        # 2. BAZA ĆWICZEŃ: WŁÓKNA I AKTYWACJA
        # 'fibers_recruited': % całkowitej puli włókien aktonu uruchamianych w ruchu
        self.EXERCISES = {
            "Flat_Barbell_Press": {
                "cat": "Push", "subcat": "compound",
                "act": {"chest_mid": 0.85, "chest_upper": 0.20, "delt_front": 0.60, "tricep_lateral": 0.70},
                "fibers_recruited": {"chest_mid": 0.90, "chest_upper": 0.30, "delt_front": 0.80, "tricep_lateral": 0.75}
            },
            "Incline_Dumbbell_Press": {
                "cat": "Push", "subcat": "compound",
                "act": {"chest_upper": 0.95, "chest_mid": 0.40, "delt_front": 0.70, "tricep_lateral": 0.50},
                "fibers_recruited": {"chest_upper": 0.95, "chest_mid": 0.50, "delt_front": 0.85, "tricep_lateral": 0.60}
            },
            "Cable_Crossover_Low_To_High": {
                "cat": "Push", "subcat": "isolation",
                "act": {"chest_upper": 0.95, "chest_mid": 0.20, "delt_front": 0.25},
                "fibers_recruited": {"chest_upper": 0.90, "chest_mid": 0.30, "delt_front": 0.30}
            },
            "Chest_Dips": {
                "cat": "Push", "subcat": "compound",
                "act": {"chest_lower": 0.95, "chest_mid": 0.50, "delt_front": 0.40, "tricep_lateral": 0.80},
                "fibers_recruited": {"chest_lower": 0.95, "chest_mid": 0.60, "delt_front": 0.50, "tricep_lateral": 0.85}
            },
            "Dumbbell_Lateral_Raises": {
                "cat": "Push", "subcat": "isolation",
                "act": {"delt_side": 0.95},
                "fibers_recruited": {"delt_side": 0.95}
            },
            # --- LEGS & CORE ---
            "Barbell_Squat": {
                "cat": "Legs", "subcat": "compound",
                "act": {"quad_vastus": 0.90, "quad_rectus": 0.65, "back_lower": 0.80, "abs_upper": 0.40},
                "fibers_recruited": {"quad_vastus": 0.95, "quad_rectus": 0.70, "back_lower": 0.90, "abs_upper": 0.50}
            },
            "Hanging_Leg_Raises": {
                "cat": "Legs", "subcat": "core",
                "act": {"abs_lower": 0.95, "abs_upper": 0.25},
                "fibers_recruited": {"abs_lower": 0.95, "abs_upper": 0.40}
            },
            "Standing_Calf_Raise": {
                "cat": "Legs", "subcat": "isolation",
                "act": {"calves": 0.95},
                "fibers_recruited": {"calves": 0.95}
            }
        }

    def _mcdonald_growth_rate(self, years):
        """Zwraca potencjał wzrostu (w gramach mięśni na tydzień) na podstawie stażu."""
        if years < 1.0: return 230.0  # ~12 kg rocznie (Początkujący - Mnożnik x12)
        if years < 2.0: return 115.0  # ~6 kg rocznie
        if years < 3.0: return 57.0  # ~3 kg rocznie
        if years < 4.0: return 28.0  # ~1.5 kg rocznie
        return 19.0  # ~1 kg rocznie (Zaawansowany)

    def _calculate_hypertrophy_yield(self, ex_data):
        """
        Oblicza potencjalny zysk masy mięśniowej w gramach z jednej serii ćwiczenia.
        Bada pulę zaangażowanych włókien i rozdziela je na "Uśpione" oraz "Zaadaptowane".
        """
        total_yield = 0.0

        for akton, recruited_pct in ex_data.get("fibers_recruited", {}).items():
            hist_eng = self.AKTONS[akton]["historical_engagement"]

            # Jak duża część włókien uderzonych tym ćwiczeniem to włókna "dziewicze"
            untapped_fibers = max(0.0, recruited_pct - hist_eng)
            adapted_fibers = min(recruited_pct, hist_eng)

            # Włókna zaadaptowane rosną z prędkością obecnego stażu (np. wolno)
            adapted_growth = adapted_fibers * self._mcdonald_growth_rate(self.stage)

            # Włókna uśpione rosną z prędkością nowicjusza (0 lat stażu)
            untapped_growth = untapped_fibers * self._mcdonald_growth_rate(0.0)

            # Mnożymy to przez właściwe napięcie (act) generowane przez ćwiczenie
            tension = ex_data["act"].get(akton, 0.0)

            akton_yield = (adapted_growth + untapped_growth) * tension
            total_yield += akton_yield

        return total_yield

    def _calculate_fatigue_cost(self, activation):
        """
        Współczynnik Ułamkowego Zmęczenia (Fractional Fatigue).
        Jeśli aktywacja jest niższa niż 0.5, seria kosztuje zaledwie ułamek budżetu.
        """
        if activation >= 0.5:
            return activation  # Pełne zmęczenie (0.5 do 1.0)
        else:
            return activation * 0.5  # Izometryczne napięcie (np. 0.4 staje się 0.2 serii MRV)

    def optimize_day(self, allowed_cat, target_fraction):
        daily_pool = {k: v for k, v in self.EXERCISES.items() if v["cat"] == allowed_cat}

        # Bramkowanie izolacji brzucha/łydek na podstawie stażu
        if self.stage < 1.5:
            daily_pool = {k: v for k, v in daily_pool.items() if
                          v["subcat"] not in ["core", "isolation" if "Calf" in k else ""]}

        if not daily_pool:
            return {}

        ex_names = list(daily_pool.keys())
        num_ex = len(ex_names)
        akton_names = list(self.AKTONS.keys())

        A_ub = np.zeros((len(akton_names) + 1, num_ex))
        b_ub = np.zeros(len(akton_names) + 1)

        # 1. Zapełnianie macierzy ograniczeń (Fractional Fatigue)
        for i, akton in enumerate(akton_names):
            mrv_adjusted = self.AKTONS[akton]["mrv"] * (1.0 + (self.stage * 0.1)) * self.recovery
            b_ub[i] = mrv_adjusted * target_fraction

            for j, ex in enumerate(ex_names):
                raw_act = daily_pool[ex]["act"].get(akton, 0.0)
                A_ub[i, j] = self._calculate_fatigue_cost(raw_act)

        A_ub[-1, :] = 1.0
        b_ub[-1] = self.MAX_SETS_SESSION

        # 2. Funkcja Celu: Maksymalizacja Hipertrofii (Zysk z włókien)
        c = np.zeros(num_ex)
        for j, ex in enumerate(ex_names):
            c[j] = -self._calculate_hypertrophy_yield(daily_pool[ex])

        bounds = [(0, 5) for _ in range(num_ex)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            plan = {}
            predicted = np.round(res.x).astype(int)
            for j, sets in enumerate(predicted):
                if sets > 0:
                    ex_name = ex_names[j]
                    est_growth = self._calculate_hypertrophy_yield(daily_pool[ex_name]) * sets
                    plan[ex_name] = {"sets": sets, "est_growth_g": round(est_growth, 1)}
            return plan
        return {}

    def generate_plan(self):
        print(f"{'=' * 70}\n HYPERTROPHY YIELD PLANNER (Współczynnik uśpionych włókien)\n{'=' * 70}")
        print(f"Staż użytkownika: {self.stage} lat | Odblokowane izolacje: {'TAK' if self.stage >= 1.5 else 'NIE'}\n")

        days = [
            {"name": "Dzień 1 (Push)", "cat": "Push"},
            {"name": "Dzień 2 (Legs & Core)", "cat": "Legs"}
        ]

        total_weekly_growth = 0.0

        for day in days:
            print(f"[ {day['name'].upper()} ]")
            day_plan = self.optimize_day(day["cat"], 0.5)

            if not day_plan:
                continue

            total_sets = sum(data["sets"] for data in day_plan.values())
            est_time = int(total_sets * self.TIME_PER_SET_MIN)
            daily_growth = sum(data["est_growth_g"] for data in day_plan.values())
            total_weekly_growth += daily_growth

            for ex, data in day_plan.items():
                print(
                    f"   -> {ex.replace('_', ' ')}: {data['sets']} serie | Przewidywany wzrost: +{data['est_growth_g']}g")

            print("-" * 50)
            print(
                f"   Objętość: {total_sets} serii | Czas: ~{est_time} min | Zysk sesji: +{round(daily_growth, 1)}g tkanki\n")

        print("=" * 70)
        print(f"ZAAWANSOWANA ESTYMACJA WZROSTU NA TEN TYDZIEŃ: +{round(total_weekly_growth, 1)} gramów")
        print("=" * 70)


if __name__ == "__main__":
    # Symulacja użytkownika o stażu 3 lata (Zaawansowany)
    planner = AdvancedHypertrophyPlanner(years_of_training=3.0, recovery_factor=1.0)
    planner.generate_plan()