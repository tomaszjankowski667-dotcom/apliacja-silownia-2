"""
ULTIMATE NEURAL PLANNER (Biomechanics + McDonald's Yield + Time-Bound)
----------------------------------------------------------------------
Kompletny system treningowy. Łączy:
1. Analizę dźwigni ze zdjęć (L_h, chest_block, L_femur).
2. Tygodniowy układ dni (PPL / Upper-Lower) i limity czasu sesji.
3. Wyliczanie zysku w gramach tkanki (model McDonalda) z podziałem na aktony.
4. Ułamkowe zmęczenie (Fractional Fatigue) dla aktywacji < 0.5.
5. Zależność od stażu (blokada izolacji brzucha/łydek dla nowicjuszy).
"""

import numpy as np
from scipy.optimize import linprog

try:
    from user_data import PROFILES
except ImportError:
    PROFILES = {
        "Brat (Z_Zdjecia)": {
            "years_of_training": 2.5, "recovery_factor": 1.0,
            "L_h": 0.326, "chest_block": 0.240, "L_femur": 0.449
        }
    }

class UltimateNeuralPlanner:
    def __init__(self, profile_name="Brat (Z_Zdjecia)"):
        self.profile_name = profile_name
        self.prof = PROFILES.get(profile_name, list(PROFILES.values())[0])

        self.stage = self.prof.get("years_of_training", 2.0)
        self.recovery = self.prof.get("recovery_factor", 1.0)

        self.TIME_PER_SET_MIN = 3.5
        self.MAX_SETS_SESSION = 22

        # BAZA AKTONÓW (MRV + Historyczne Zaangażowanie 0.0-1.0)
        self.AKTONS = {
            "chest_upper": {"mrv": 12.0, "hist": 0.30},
            "chest_mid": {"mrv": 14.0, "hist": 0.95},
            "chest_lower": {"mrv": 10.0, "hist": 0.15},
            "delt_front": {"mrv": 8.0, "hist": 0.95},
            "delt_side": {"mrv": 16.0, "hist": 0.40},
            "delt_rear": {"mrv": 14.0, "hist": 0.20},
            "tricep_long": {"mrv": 10.0, "hist": 0.30},
            "tricep_lateral": {"mrv": 12.0, "hist": 0.85},
            "bicep_long": {"mrv": 10.0, "hist": 0.80},
            "bicep_short": {"mrv": 12.0, "hist": 0.50},
            "abs_upper": {"mrv": 10.0, "hist": 0.80},
            "abs_lower": {"mrv": 10.0, "hist": 0.20},
            "abs_oblique": {"mrv": 8.0, "hist": 0.10},
            "quad_vastus": {"mrv": 14.0, "hist": 0.90},
            "quad_rectus": {"mrv": 12.0, "hist": 0.50},
            "hamstring": {"mrv": 12.0, "hist": 0.40},
            "calves": {"mrv": 14.0, "hist": 0.10},
            "back_lats": {"mrv": 14.0, "hist": 0.85},
            "back_mid": {"mrv": 14.0, "hist": 0.40},
            "back_lower": {"mrv": 10.0, "hist": 0.60}
        }

    def _calc_personal_exercise_efficiency(self):
        l_h = self.prof.get("L_h", 0.326)
        chest_block = self.prof.get("chest_block", 0.240)
        l_femur = self.prof.get("L_femur", 0.449)

        barbell_arm_penalty = 1.0 if (chest_block - (l_h * 0.8)) > 0 else 0.75
        squat_femur_penalty = 0.80 if l_femur > 0.43 else 1.0

        database = {
            "Flat_Barbell_Press": {
                "cat": "Push", "subcat": "compound",
                "act": {"chest_mid": 0.85 * barbell_arm_penalty, "chest_upper": 0.20, "delt_front": 0.60, "tricep_lateral": 0.70},
                "fibers": {"chest_mid": 0.90, "chest_upper": 0.30, "delt_front": 0.80, "tricep_lateral": 0.75}
            },
            "Incline_Dumbbell_Press": {
                "cat": "Push", "subcat": "compound",
                "act": {"chest_upper": 0.95, "chest_mid": 0.40, "delt_front": 0.70, "tricep_lateral": 0.50},
                "fibers": {"chest_upper": 0.95, "chest_mid": 0.50, "delt_front": 0.85, "tricep_lateral": 0.60}
            },
            "Cable_Crossover_Low_To_High": {
                "cat": "Push", "subcat": "isolation",
                "act": {"chest_upper": 0.95, "chest_mid": 0.20, "delt_front": 0.25},
                "fibers": {"chest_upper": 0.90, "chest_mid": 0.30, "delt_front": 0.30}
            },
            "Chest_Dips": {
                "cat": "Push", "subcat": "compound",
                "act": {"chest_lower": 0.95, "chest_mid": 0.50, "delt_front": 0.40, "tricep_lateral": 0.80},
                "fibers": {"chest_lower": 0.95, "chest_mid": 0.60, "delt_front": 0.50, "tricep_lateral": 0.85}
            },
            "Dumbbell_Lateral_Raises": {
                "cat": "Push", "subcat": "isolation",
                "act": {"delt_side": 0.95, "delt_front": 0.10},
                "fibers": {"delt_side": 0.95}
            },
            "Pull_Ups": {
                "cat": "Pull", "subcat": "compound",
                "act": {"back_lats": 0.95, "back_mid": 0.40, "bicep_short": 0.60},
                "fibers": {"back_lats": 0.95, "back_mid": 0.50, "bicep_short": 0.70}
            },
            "Barbell_Row": {
                "cat": "Pull", "subcat": "compound",
                "act": {"back_mid": 0.90, "back_lats": 0.50, "back_lower": 0.60, "delt_rear": 0.40, "bicep_long": 0.50},
                "fibers": {"back_mid": 0.95, "back_lats": 0.60, "back_lower": 0.80, "delt_rear": 0.60, "bicep_long": 0.70}
            },
            "Reverse_Pec_Deck": {
                "cat": "Pull", "subcat": "isolation",
                "act": {"delt_rear": 0.95, "back_mid": 0.30},
                "fibers": {"delt_rear": 0.95, "back_mid": 0.40}
            },
            "Barbell_Squat": {
                "cat": "Legs", "subcat": "compound",
                "act": {"quad_vastus": 0.90 * squat_femur_penalty, "quad_rectus": 0.65, "back_lower": 0.80, "abs_upper": 0.40},
                "fibers": {"quad_vastus": 0.95, "quad_rectus": 0.70, "back_lower": 0.90, "abs_upper": 0.50}
            },
            "Romanian_Deadlift": {
                "cat": "Legs", "subcat": "compound",
                "act": {"hamstring": 0.95, "back_lower": 0.85, "abs_upper": 0.30},
                "fibers": {"hamstring": 0.95, "back_lower": 0.95, "abs_upper": 0.40}
            },
            "Cable_Crunches": {
                "cat": "Legs", "subcat": "core",
                "act": {"abs_upper": 0.95, "abs_lower": 0.30},
                "fibers": {"abs_upper": 0.95, "abs_lower": 0.40}
            },
            "Hanging_Leg_Raises": {
                "cat": "Legs", "subcat": "core",
                "act": {"abs_lower": 0.95, "abs_upper": 0.25},
                "fibers": {"abs_lower": 0.95, "abs_upper": 0.40}
            },
            "Standing_Calf_Raise": {
                "cat": "Legs", "subcat": "isolation",
                "act": {"calves": 0.95},
                "fibers": {"calves": 0.95}
            }
        }
        return database

    def _mcdonald_growth_rate_per_akton(self, years):
        """Zwraca potencjał przyrostu na JEDEN AKTON w gramach tygodniowo (całe ciało = 20 aktonów)."""
        num_aktons = len(self.AKTONS)
        if years < 1.0: return 230.0 / num_aktons   # ~11.5g na akton / tydzień
        if years < 2.0: return 115.0 / num_aktons   # ~5.75g na akton
        if years < 3.0: return 57.0 / num_aktons    # ~2.85g na akton
        if years < 4.0: return 28.0 / num_aktons    # ~1.40g na akton
        return 19.0 / num_aktons                    # ~0.95g na akton

    def _calc_hypertrophy_yield(self, ex_data):
        """Wylicza potencjalny zysk masy z JEDNEJ SERII ćwiczenia dla przydzielonych włókien."""
        total_yield_per_set = 0.0

        for akton, recruited_pct in ex_data.get("fibers", {}).items():
            hist_eng = self.AKTONS[akton]["hist"]
            mrv = self.AKTONS[akton]["mrv"]  # Pojemność hipertroficzna aktonu

            untapped = max(0.0, recruited_pct - hist_eng)
            adapted = min(recruited_pct, hist_eng)

            # Wartość 1 serii dla tego aktonu = (Maksymalny zysk tygodniowy / MRV) * % zaangażowania
            adapted_growth_per_set = (self._mcdonald_growth_rate_per_akton(self.stage) / mrv) * adapted
            untapped_growth_per_set = (self._mcdonald_growth_rate_per_akton(0.0) / mrv) * untapped

            tension = ex_data["act"].get(akton, 0.0)

            # Dodajemy przewidywany zysk tkanki z jednej serii
            total_yield_per_set += (adapted_growth_per_set + untapped_growth_per_set) * tension

        return total_yield_per_set

    def _calc_fatigue_cost(self, activation):
        return activation if activation >= 0.5 else activation * 0.5

    def get_split_structure(self):
        if self.stage < 1.0:
            return "Upper_Lower", [
                {"name": "Dzień 1 (Góra)", "cats": ["Push", "Pull"], "fraction": 0.5},
                {"name": "Dzień 2 (Dół + Brzuch)", "cats": ["Legs"], "fraction": 0.5},
                {"name": "Dzień 3 (Rest)", "cats": [], "fraction": 0.0},
                {"name": "Dzień 4 (Góra)", "cats": ["Push", "Pull"], "fraction": 0.5},
                {"name": "Dzień 5 (Dół + Brzuch)", "cats": ["Legs"], "fraction": 0.5},
                {"name": "Dzień 6 (Rest)", "cats": [], "fraction": 0.0},
                {"name": "Dzień 7 (Rest)", "cats": [], "fraction": 0.0},
            ]
        else:
            return "PPL", [
                {"name": "Dzień 1 (Push)", "cats": ["Push"], "fraction": 0.5},
                {"name": "Dzień 2 (Pull)", "cats": ["Pull"], "fraction": 0.5},
                {"name": "Dzień 3 (Legs & Core)", "cats": ["Legs"], "fraction": 0.5},
                {"name": "Dzień 4 (Rest)", "cats": [], "fraction": 0.0},
                {"name": "Dzień 5 (Push)", "cats": ["Push"], "fraction": 0.5},
                {"name": "Dzień 6 (Pull)", "cats": ["Pull"], "fraction": 0.5},
                {"name": "Dzień 7 (Legs & Core)", "cats": ["Legs"], "fraction": 0.5},
            ]

    def optimize_day(self, db_exercises, allowed_cats, mrv_fraction):
        daily_pool = {k: v for k, v in db_exercises.items() if v["cat"] in allowed_cats}

        if self.stage < 1.5:
            daily_pool = {k: v for k, v in daily_pool.items() if v["subcat"] not in ["core", "isolation" if "Calf" in k else ""]}

        if not daily_pool:
            return {}

        ex_names = list(daily_pool.keys())
        num_ex = len(ex_names)
        akton_names = list(self.AKTONS.keys())

        A_ub = np.zeros((len(akton_names) + 1, num_ex))
        b_ub = np.zeros(len(akton_names) + 1)

        mrv_multiplier = (1.0 + (self.stage * 0.15)) * self.recovery

        for i, akton in enumerate(akton_names):
            b_ub[i] = self.AKTONS[akton]["mrv"] * mrv_multiplier * mrv_fraction
            for j, ex in enumerate(ex_names):
                raw_act = daily_pool[ex]["act"].get(akton, 0.0)
                A_ub[i, j] = self._calc_fatigue_cost(raw_act)

        A_ub[-1, :] = 1.0
        b_ub[-1] = self.MAX_SETS_SESSION

        c = np.zeros(num_ex)
        for j, ex in enumerate(ex_names):
            c[j] = -self._calc_hypertrophy_yield(daily_pool[ex])

        bounds = []
        for ex in ex_names:
            if daily_pool[ex]["subcat"] == "core":
                bounds.append((2, 4))
            elif "Calf" in ex:
                bounds.append((3, 5))
            else:
                bounds.append((0, 5))

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            plan = {}
            predicted = np.round(res.x).astype(int)
            for j, sets in enumerate(predicted):
                if sets > 0:
                    ex_name = ex_names[j]
                    est_growth = self._calc_hypertrophy_yield(daily_pool[ex_name]) * sets
                    plan[ex_name] = {"sets": sets, "growth_g": est_growth}
            return plan
        return {}

    def generate_full_schedule(self):
        db_exercises = self._calc_personal_exercise_efficiency()
        split_name, schedule = self.get_split_structure()

        print(f"\n{'='*70}")
        print(f" ULTIMATE NEURAL PLANNER (Profil: {self.profile_name})")
        print(f" Staż: {self.stage} lat | Split: {split_name} | Izolacja Core/Łydek: {'TAK' if self.stage >= 1.5 else 'NIE'}")
        print(f"{'='*70}")

        total_weekly_growth = 0.0

        for day in schedule:
            print(f"\n[ {day['name'].upper()} ]")

            if not day["cats"]:
                print("   Odpoczynek układu nerwowego i superkompensacja.")
                continue

            day_plan = self.optimize_day(db_exercises, day["cats"], day["fraction"])

            if not day_plan:
                continue

            total_sets = sum(data["sets"] for data in day_plan.values())
            est_time = int(total_sets * self.TIME_PER_SET_MIN)
            daily_growth = sum(data["growth_g"] for data in day_plan.values())
            total_weekly_growth += daily_growth

            for ex, data in day_plan.items():
                clean_name = ex.replace("_", " ")
                print(f"   -> {clean_name}: {data['sets']} serie (Szac. wzrost: +{round(data['growth_g'], 1)}g)")

            print("-" * 50)
            print(f"   Objętość: {total_sets} serii | Czas: ~{est_time} min | Zysk sesji: +{round(daily_growth, 1)}g")

        print(f"\n{'='*70}")
        print(f" MIESIĘCZNA PROGNOZA WZROSTU TKANKI MIĘŚNIOWEJ: +{round(total_weekly_growth * 4.33, 1)} gramów")
        print(f"{'='*70}")

if __name__ == "__main__":
    planner = UltimateNeuralPlanner("Brat (Z_Zdjecia)")
    planner.generate_full_schedule()