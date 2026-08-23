"""
UNIVERSAL DIET BRIDGE (universal_diet_bridge.py)
------------------------------------------------
Uniwersalny moduł integrujący dowolne źródło danych dietetycznych:
- Apple Health / Google Health Connect (Standard uniwersalny)
- Bezpośrednie API partnerskie (np. Fitatu / B2B)
- Ręczny tryb użytkownika (Fallback, gdy ktoś nie łączy aplikacji)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional


# --- 1. ZUNIFIKOWANY MODEL DANYCH (Standard wyjściowy dla silnika treningowego) ---

class StandardDietData:
    def __init__(self, calories_consumed: float, protein_g: float, carbs_g: float, fats_g: float, source: str):
        self.calories_consumed = calories_consumed
        self.protein_g = protein_g
        self.carbs_g = carbs_g
        self.fats_g = fats_g
        self.source = source
        self.timestamp = datetime.now().isoformat()

    def determine_energy_phase(self, calculated_tdee: float) -> str:
        """
        Ocenia bilans kaloryczny na podstawie porównania ze spalaniem TDEE:
        - DEFICIT (Ostra redukcja -> tniemy objętość treningową)
        - MAINTENANCE (Utrzymanie -> standardowa objętość)
        - SURPLUS (Masa / Nadwyżka -> podbijamy objętość MRV)
        """
        diff = self.calories_consumed - calculated_tdee
        if diff < -350:
            return "DEFICIT"
        elif diff > 250:
            return "SURPLUS"
        else:
            return "MAINTENANCE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calories": self.calories_consumed,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fats_g": self.fats_g,
            "source": self.source
        }


# --- 2. BAZOWY INTERFEJS ADAPTERA ---

class BaseDietAdapter(ABC):
    @abstractmethod
    def fetch_daily_nutrition(self, user_identifier: str) -> Optional[StandardDietData]:
        """Pobiera i normalizuje dane do standardu StandardDietData."""
        pass


# --- 3. IMPLEMENTACJE DLA RÓŻNYCH APLIKACJI I ŹRÓDEŁ ---

class HealthConnectAdapter(BaseDietAdapter):
    """
    Uniwersalny adapter pod Apple Health / Google Health Connect.
    Działa automatycznie z Fitatu, MyFitnessPal, Yazio, Cronometer itp.
    """

    def fetch_daily_nutrition(self, user_identifier: str) -> Optional[StandardDietData]:
        # Symulacja odczytu ustandaryzowanego rekordu z Apple HealthKit / Health Connect
        payload_from_os = {
            "total_kcal": 2450.0,
            "protein": 165.0,
            "carbohydrates": 280.0,
            "fat": 70.0
        }
        return StandardDietData(
            calories_consumed=payload_from_os["total_kcal"],
            protein_g=payload_from_os["protein"],
            carbs_g=payload_from_os["carbohydrates"],
            fats_g=payload_from_os["fat"],
            source="Apple Health / Google Health Connect (Auto-Sync)"
        )


class DirectFitatuAdapter(BaseDietAdapter):
    """
    Adapter dedykowany pod bezpośrednie API Fitatu (gdybyśmy mieli umowę partnerską B2B).
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_daily_nutrition(self, user_identifier: str) -> Optional[StandardDietData]:
        # Symulacja specyficznego formatu JSON z API Fitatu
        fitatu_raw_response = {
            "fitatu_user": user_identifier,
            "summary_energy_kcal": 2600.0,
            "macro_protein": 175.0,
            "macro_carbs": 310.0,
            "macro_fat": 72.0
        }
        return StandardDietData(
            calories_consumed=fitatu_raw_response["summary_energy_kcal"],
            protein_g=fitatu_raw_response["macro_protein"],
            carbs_g=fitatu_raw_response["macro_carbs"],
            fats_g=fitatu_raw_response["macro_fat"],
            source="Direct Fitatu API"
        )


class ManualInputAdapter(BaseDietAdapter):
    """
    Dla użytkowników, którzy nie korzystają z żadnej aplikacji
    i chcą po prostu wybrać swój cel (np. suwak w aplikacji: 'Jestem na redukcji').
    """

    def __init__(self, declared_calories: float, declared_protein: float):
        self.declared_calories = declared_calories
        self.declared_protein = declared_protein

    def fetch_daily_nutrition(self, user_identifier: str) -> Optional[StandardDietData]:
        return StandardDietData(
            calories_consumed=self.declared_calories,
            protein_g=self.declared_protein,
            carbs_g=0.0,  # Brak szczegółów
            fats_g=0.0,
            source="Manual User Input"
        )


# --- 4. GŁÓWNY ZARZĄDCA (Diet Manager) ---

class UniversalDietManager:
    def __init__(self, adapter: BaseDietAdapter):
        self.adapter = adapter

    def get_nutrition_report_for_training(self, user_id: str, user_tdee: float) -> Dict[str, Any]:
        """
        Zwraca pełen raport gotowy do przekazania do Auto-Programmera.
        """
        nutrition_data = self.adapter.fetch_daily_nutrition(user_id)
        if not nutrition_data:
            return {"status": "UNKNOWN", "phase": "MAINTENANCE", "volume_modifier": 1.0}

        energy_phase = nutrition_data.determine_energy_phase(user_tdee)

        # Modyfikator objętości serii treningowych zależny od bilansu kalorycznego
        if energy_phase == "DEFICIT":
            volume_modifier = 0.75  # Ścinamy 25% objętości (ochrona CNS i stawów)
            advice = "Wykryto deficyt kaloryczny. Objętość treningowa została zredukowana do poziomu MEV."
        elif energy_phase == "SURPLUS":
            volume_modifier = 1.20  # Dodajemy 20% objętości (nadwyżka = świetna regeneracja)
            advice = "Wykryto nadwyżkę kaloryczną. Algorytm zwiększył objętość serii stymulujących (MRV)."
        else:
            volume_modifier = 1.00
            advice = "Kalorie na poziomie zapotrzebowania (Zero metaboliczne). Standardowa progresja."

        return {
            "source": nutrition_data.source,
            "nutrition": nutrition_data.to_dict(),
            "energy_phase": energy_phase,
            "volume_modifier": volume_modifier,
            "coach_advice": advice
        }


# --- DEMONSTRACJA DZIAŁANIA ---

if __name__ == "__main__":
    USER_TDEE = 2800.0  # Wyliczone przez nasz silnik zapotrzebowanie (BMR + trening)

    print("=" * 65)
    print("TEST UNIWERSALNEGO MOSTU DIETETYCZNEGO")
    print("=" * 65)

    # Scenariusz A: Użytkownik ma podpięte Apple Health / Google Health Connect
    health_sync = UniversalDietManager(adapter=HealthConnectAdapter())
    report_a = health_sync.get_nutrition_report_for_training(user_id="user_101", user_tdee=USER_TDEE)

    print(f"\n[SCENARIUSZ A: Integracja Systemowa ({report_a['source']})]")
    print(f"  * Spożyte kalorie: {report_a['nutrition']['calories']} kcal vs TDEE: {USER_TDEE} kcal")
    print(f"  * Faza energetyczna: {report_a['energy_phase']}")
    print(f"  * Modyfikator objętości treningu: {report_a['volume_modifier']}x")
    print(f"  * Wskazówka: {report_a['coach_advice']}")

    # Scenariusz B: Użytkownik ręcznie wpisał, że je 3300 kcal (Masa)
    manual_sync = UniversalDietManager(adapter=ManualInputAdapter(declared_calories=3300, declared_protein=180))
    report_b = manual_sync.get_nutrition_report_for_training(user_id="user_101", user_tdee=USER_TDEE)

    print(f"\n[SCENARIUSZ B: Wpis Ręczny ({report_b['source']})]")
    print(f"  * Spożyte kalorie: {report_b['nutrition']['calories']} kcal vs TDEE: {USER_TDEE} kcal")
    print(f"  * Faza energetyczna: {report_b['energy_phase']}")
    print(f"  * Modyfikator objętości treningu: {report_b['volume_modifier']}x")
    print(f"  * Wskazówka: {report_b['coach_advice']}")
    print("=" * 65)