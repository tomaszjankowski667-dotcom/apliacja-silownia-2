"""
APP ORCHESTRATOR & DYNAMIC AUTO-PROGRAMMER (dynamic_auto_programmer.py)
-----------------------------------------------------------------------
Główny mózg aplikacji (Klej systemowy).
Nie układa planu sam z siebie, lecz komunikuje się z pozostałymi modułami:
1. Pobiera profil (user_data.py)
2. Sprawdza sprzęt (smart_equipment_manager.py)
3. Zlicza zmęczenie (fatigue_volume_tracker.py)
4. Wyciąga fizykę (biomechanical_bioscore_engine.py -> folder analyze)
5. Zleca ułożenie planu (planner/global_training_planner.py)
"""

import sys
import logging
from user_data import PROFILES
from planner.global_training_planner import UltimateNeuralPlanner

# --- IMPORTY MIKROSERWISÓW (Zabezpieczone przed brakiem kodu) ---
try:
    from fatigue_volume_tracker import FatigueTracker
except ImportError:
    FatigueTracker = None

try:
    from smart_equipment_manager import EquipmentManager
except ImportError:
    EquipmentManager = None

try:
    from biomechanical_bioscore_engine import BiomechanicalEngine
except ImportError:
    BiomechanicalEngine = None


class DynamicAutoProgrammer:
    def __init__(self, profile_name="Brat (Z_Zdjecia)"):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.profile_name = profile_name

        # 1. Baza Danych: Pobranie anatomii
        if self.profile_name not in PROFILES:
            logging.error(f"Nie znaleziono profilu: {self.profile_name}")
            sys.exit(1)

        self.user_anatomy = PROFILES[self.profile_name]
        self.stage = self.user_anatomy.get("years_of_training", 1.0)

        logging.info(f"Zainicjowano sesję dla: {self.profile_name} (Staż: {self.stage} lat)")

    def _sync_equipment(self):
        """Krok 1: Komunikacja z menedżerem sprzętu."""
        if EquipmentManager:
            manager = EquipmentManager(self.profile_name)
            # Zwraca np. ["Dumbbells", "Barbell", "Bodyweight"] i odrzuca maszyny
            return manager.get_available_equipment()
        logging.warning("Brak modułu smart_equipment_manager.py - zakładam pełny dostęp do siłowni.")
        return ["ALL"]

    def _sync_fatigue_and_recovery(self):
        """Krok 2: Komunikacja z trackerem zmęczenia (Złote proporcje uśpionych włókien)."""
        if FatigueTracker:
            tracker = FatigueTracker(self.profile_name)
            # Pobiera historię treningów, by wskazać które włókna to "nowicjusze"
            historical_engagement = tracker.get_historical_akton_engagement()
            current_mrv_modifier = tracker.get_cns_fatigue_modifier()
            return historical_engagement, current_mrv_modifier

        logging.warning("Brak modułu fatigue_volume_tracker.py - ładuję bazowe MRV.")
        return None, 1.0

    def _sync_biomechanics(self, available_eq):
        """Krok 3: Komunikacja z silnikiem biomechanicznym (folder /analyze)."""
        if BiomechanicalEngine:
            engine = BiomechanicalEngine(self.user_anatomy)
            # Silnik przepuszcza wszystkie pliki analyze_*.py i wypluwa tylko te ćwiczenia,
            # które pasują do dźwigni użytkownika i dostępnego sprzętu
            return engine.evaluate_all_exercises(equipment_filter=available_eq)

        logging.warning("Brak modułu biomechanical_bioscore_engine.py - ładuję domyślną bazę planera.")
        return None

    def build_and_deploy_program(self):
        """Krok 4: Główny pipeline - spięcie danych i przekazanie do Planera."""
        print("\n" + "=" * 60)
        print(" URUCHAMIANIE DYNAMIC AUTO-PROGRAMMER (ORCHESTRATOR)")
        print("=" * 60)

        # Pobieranie danych z mikroserwisów
        equipment = self._sync_equipment()
        history, fatigue_modifier = self._sync_fatigue_and_recovery()
        custom_exercises_db = self._sync_biomechanics(equipment)

        # Inicjalizacja Głównego Planera z dynamicznymi danymi
        planner = UltimateNeuralPlanner(profile_name=self.profile_name)

        # Wstrzyknięcie zebranych zmiennych do instancji planera (Nadpisywanie twardych danych)
        planner.recovery = planner.recovery * fatigue_modifier

        if history:
            # Aktualizacja historycznego zaangażowania aktonów
            for akton, data in history.items():
                if akton in planner.AKTONS:
                    planner.AKTONS[akton]["hist"] = data

        if custom_exercises_db:
            # Podmiana bazy na przefiltrowaną przez biomechanikę
            pass # Docelowo: planner.EXERCISES = custom_exercises_db

        logging.info("Dane zsynchronizowane. Przekazuję kontrolę do UltimateNeuralPlanner...\n")

        # Ostateczne wygenerowanie planu
        planner.generate_full_schedule()


if __name__ == "__main__":
    # Punkt startowy dla całego backendu
    orchestrator = DynamicAutoProgrammer(profile_name="Brat (Z_Zdjecia)")
    orchestrator.build_and_deploy_program()