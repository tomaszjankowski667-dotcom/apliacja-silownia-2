"""
CHEST NEURAL PLANNER & BIOMECHANICAL OPTIMIZER
----------------------------------------------
Moduł dobierający optymalny plan treningowy dla klatki piersiowej.
Uwzględnia fizjologiczny rozkład włókien (PCSA), zmęczenie (MRV)
oraz trenuje sieć neuronową do szybkiej inferencji planów.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linprog

# --- 1. BAZA DANYCH ĆWICZEŃ: MACIERZ ZACHODZENIA I KOSZT ZMĘCZENIA ---
# [Zysk_Góra, Zysk_Środek, Zysk_Dół], Koszt_CNS_na_serię (w skali 1.0 - 5.0)
EXERCISE_DATABASE = {
    "Incline_Dumbbell_Press": {
        "activation": np.array([0.85, 0.40, 0.05]),
        "fatigue_cost": 2.8
    },
    "Incline_Smith_Press": {
        "activation": np.array([0.90, 0.35, 0.05]),
        "fatigue_cost": 2.3
    },
    "Low_to_High_Cable_Fly": {
        "activation": np.array([0.95, 0.20, 0.00]),
        "fatigue_cost": 1.2
    },
    "Flat_Barbell_Bench_Press": {
        "activation": np.array([0.25, 0.85, 0.25]),
        "fatigue_cost": 4.5
    },
    "Flat_Dumbbell_Press": {
        "activation": np.array([0.40, 0.80, 0.20]),
        "fatigue_cost": 3.0
    },
    "Pec_Deck_Fly": {
        "activation": np.array([0.15, 0.95, 0.15]),
        "fatigue_cost": 1.4
    },
    "Chest_Dips": {
        "activation": np.array([0.10, 0.50, 0.90]),
        "fatigue_cost": 3.8
    },
    "High_to_Low_Cable_Fly": {
        "activation": np.array([0.05, 0.30, 0.95]),
        "fatigue_cost": 1.3
    }
}

EXERCISE_NAMES = list(EXERCISE_DATABASE.keys())
NUM_EXERCISES = len(EXERCISE_NAMES)


# --- 2. SOLVER MATEMATYCZNY (GROUND TRUTH GENERATOR) ---
class ChestOptimizationEngine:
    @staticmethod
    def solve_workout(priority="upper", user_mrv=30.0, max_sets_per_ex=4):
        """
        Rozwiązuje problem programowania liniowego, dobierając liczby serii.
        """
        # Wagi celów estetycznych (Góra, Środek, Dół)
        if priority == "upper":
            target_weights = np.array([0.55, 0.30, 0.15])
        elif priority == "lower":
            target_weights = np.array([0.25, 0.35, 0.40])
        else:  # balanced (Złote Proporcje)
            target_weights = np.array([0.40, 0.40, 0.20])

        # Wektor zysku dla każdego ćwiczenia po przemnożeniu przez wagi celu
        c = []
        fatigue_vector = []
        for name in EXERCISE_NAMES:
            act = EXERCISE_DATABASE[name]["activation"]
            gain = np.dot(act, target_weights)
            c.append(-gain)  # linprog minimalizuje funkcję celu, więc dajemy minus
            fatigue_vector.append(EXERCISE_DATABASE[name]["fatigue_cost"])

        # Ograniczenia: Suma zmęczenia <= user_mrv
        A_ub = [fatigue_vector]
        b_ub = [user_mrv]

        # Granice na serie dla każdego ćwiczenia (0 do max_sets_per_ex)
        bounds = [(0, max_sets_per_ex) for _ in range(NUM_EXERCISES)]

        # Rozwiązanie optymalizacyjne
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            # Zaokrąglanie do pełnych serii
            sets_distribution = np.round(res.x).astype(int)
            return sets_distribution
        else:
            return np.zeros(NUM_EXERCISES, dtype=int)


# --- 3. SIEĆ NEURONOWA UCZĄCA SIĘ ROZSZERZANIA WZORCÓW TRENINGOWYCH ---
class ChestPolicyNetwork(nn.Module):
    def __init__(self, input_dim=5, output_dim=NUM_EXERCISES):
        super(ChestPolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()  # Liczba serii nie może być ujemna
        )

    def forward(self, x):
        return self.net(x)


# --- 4. PIPELINE GENEROWANIA DANYCH I TRENOWANIA SIECI ---
def train_neural_planner(epochs=400):
    print("1. Generowanie danych syntetycznych za pomocą solvera biomechanicznego...")

    X_train = []
    y_train = []

    priority_map = {"upper": [1, 0, 0], "balanced": [0, 1, 0], "lower": [0, 0, 1]}

    # Generujemy 2000 różnych scenariuszy użytkowników
    for _ in range(2000):
        # Parametry wejściowe: [Priorytet_OneHot (3), Staż (0.1 - 1.0), MRV (15.0 - 45.0)]
        p_choice = np.random.choice(["upper", "balanced", "lower"])
        p_vec = priority_map[p_choice]
        stage = np.random.uniform(0.1, 1.0)
        mrv = np.random.uniform(15.0, 45.0)

        feature_vector = p_vec + [stage, mrv]

        # Wyliczenie idealnego planu przez solver
        optimal_sets = ChestOptimizationEngine.solve_workout(
            priority=p_choice,
            user_mrv=mrv,
            max_sets_per_ex=4
        )

        X_train.append(feature_vector)
        y_train.append(optimal_sets)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    print("2. Rozpoczynanie treningu sieci neuronowej (ChestPolicyNetwork)...")
    model = ChestPolicyNetwork(input_dim=5, output_dim=NUM_EXERCISES)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"   Epoka [{epoch + 1}/{epochs}] | Błąd MSE: {loss.item():.4f}")

    print("3. Trening zakończony sukcesem.\n")
    return model


# --- 5. TEST PRAKTYCZNY SIECI DLA BRATA ---
if __name__ == "__main__":
    # Wytrenowanie sieci
    model = train_neural_planner()

    # Test: Brat (Priorytet: Góra klatki, Staż: Średniozaawansowany 0.6, Budżet Zmęczenia MRV: 28.0)
    # Wejście: [Upper(1), Balanced(0), Lower(0), Stage(0.6), MRV(28.0)]
    test_input = torch.tensor([[1.0, 0.0, 0.0, 0.6, 28.0]], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_sets = model(test_input).numpy()[0]
        final_sets = np.round(predicted_sets).astype(int)

    print("=" * 60)
    print("WYGENEROWANY PLAN DLA PROFILU: BRAT (PRIORYTET GÓRA KLATKI)")
    print("=" * 60)

    total_cns_fatigue = 0
    total_sets = 0

    for i, ex_name in enumerate(EXERCISE_NAMES):
        sets = final_sets[i]
        if sets > 0:
            cost = EXERCISE_DATABASE[ex_name]["fatigue_cost"] * sets
            total_cns_fatigue += cost
            total_sets += sets
            act = EXERCISE_DATABASE[ex_name]["activation"]
            print(f"-> {ex_name.replace('_', ' ')}: {sets} serie")
            print(f"   Aktywacja [Góra/Środek/Dół]: {act} | Koszt CNS: {cost:.1f}")

    print("-" * 60)
    print(f"Łączna liczba serii: {total_sets}")
    print(f"Wykorzystany budżet zmęczenia (MRV): {total_cns_fatigue:.1f} / 28.0")
    print("=" * 60)