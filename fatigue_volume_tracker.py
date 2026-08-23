"""
FATIGUE & VOLUME TRACKER (fatigue_volume_tracker.py)
----------------------------------------------------
Moduł śledzący zmęczenie układu nerwowego (CNS) i lokalnego (mięśnie).
Integruje:
1. Jakość techniki z AI (Form Quality z CV)
2. Raport dietetyczny (Faza deficytu/nadwyżki)
3. Subiektywne odczucie i ciężar (RPE)
Zwraca status gotowości na następny trening i modyfikator objętości.
"""

import math
import numpy as np


class FatigueTracker:
    def __init__(self):
        # Maksymalny próg zmęczenia CNS (0-100)
        self.MAX_FATIGUE_SCORE = 100.0
        self.DELOAD_THRESHOLD = 85.0
        self.MAINTENANCE_THRESHOLD = 60.0

    def calculate_set_cns_tax(self, rpe, form_quality, is_compound):
        """
        Oblicza koszt neurologiczny (CNS Tax) pojedynczej serii.
        Ćwiczenia wielostawowe i słaba technika drastycznie podnoszą koszt.
        """
        # Baza zmęczenia rośnie wykładniczo (RPE 10 męczy nieporównywalnie mocniej niż RPE 8)
        base_fatigue = math.exp(rpe * 0.4)

        # Mnożnik za wielostawowość (Przysiad obciąża CNS znacznie bardziej niż izolacja bicepsa)
        compound_multiplier = 1.6 if is_compound else 0.7

        # Kara za słabą technikę (Szarpanie ciężaru lub brak pauzy uszkadza tkanki łączące i CNS)
        # Jeśli CV AI oceniło form_quality poniżej 1.0, zmęczenie rośnie
        technique_penalty = 1.0 + (1.0 - form_quality) if form_quality < 1.0 else 0.85

        return base_fatigue * compound_multiplier * technique_penalty

    def analyze_readiness(self, recent_sets, diet_modifier, sleep_quality=1.0):
        """
        Ocenia gotowość (Readiness) do kolejnego treningu w skali 1-100.

        :param recent_sets: Lista słowników serii z ostatniego treningu
        :param diet_modifier: Współczynnik z universal_diet_bridge (np. 0.75 w deficycie)
        :param sleep_quality: Ocena snu (np. 0.6 - źle, 1.0 - normalnie, 1.2 - świetnie)
        """
        total_cns_tax = 0.0

        for s in recent_sets:
            tax = self.calculate_set_cns_tax(s["rpe"], s["form_quality"], s["is_compound"])
            total_cns_tax += tax

        # Zdolność do regeneracji zależy od diety i snu
        recovery_capacity = diet_modifier * sleep_quality

        # Jeśli regeneracja jest zaburzona (np. brak snu + deficyt kcal), odczuwalne zmęczenie rośnie
        adjusted_fatigue = (total_cns_tax / recovery_capacity) if recovery_capacity > 0 else (total_cns_tax * 2)

        # Normalizacja do skali 1-100 (stała arbitralna do demonstracji)
        fatigue_score = min(self.MAX_FATIGUE_SCORE, adjusted_fatigue / 15.0)

        # Algorytm podejmuje decyzję o następnym cyklu:
        if fatigue_score >= self.DELOAD_THRESHOLD:
            action = "AUTO-DELOAD"
            recommendation = "Krytyczne przeciążenie CNS. Zejdź z objętości roboczej o 50% lub zrób 2 dni całkowitego odpoczynku."
            volume_adjustment = 0.50

        elif fatigue_score >= self.MAINTENANCE_THRESHOLD:
            action = "MAINTENANCE (MEV)"
            recommendation = "Podwyższone zmęczenie. Utrzymaj obecną objętość, nie dokładaj serii ani obciążeń na siłę."
            volume_adjustment = 0.90

        else:
            action = "PROGRESSION (MRV)"
            recommendation = "System zgłasza świetną regenerację. Układ nerwowy jest gotowy na progresję (dodanie ciężaru lub 1-2 serii)."
            volume_adjustment = 1.15

        return {
            "cns_fatigue_score": round(fatigue_score, 1),
            "status": action,
            "coach_recommendation": recommendation,
            "next_session_volume_multiplier": volume_adjustment
        }


# --- SYMULACJA DZIAŁANIA ---

if __name__ == "__main__":
    tracker = FatigueTracker()

    print("=" * 70)
    print("TEST MODUŁU ŚLEDZENIA ZMĘCZENIA (CNS FATIGUE TRACKER)")
    print("=" * 70)

    # Scenariusz 1: Ciężki trening nóg (wielostawowy), słaba technika, na redukcji (deficyt) i po złym śnie.
    hard_leg_session = [
        {"rpe": 9.5, "form_quality": 0.70, "is_compound": True},  # Szarpany Hack Przysiad blisko upadku
        {"rpe": 10.0, "form_quality": 0.65, "is_compound": True},  # RDL z utratą techniki na maxa
        {"rpe": 9.0, "form_quality": 0.85, "is_compound": False}  # Uginanie nóg
    ]

    report_bad_recovery = tracker.analyze_readiness(
        recent_sets=hard_leg_session,
        diet_modifier=0.75,  # Deficyt kaloryczny (z Universal Diet Bridge)
        sleep_quality=0.70  # Słaby sen
    )

    print("\n[SCENARIUSZ A: 'Ego Lifting' na Deficycie Kalorycznym]")
    print(f"  * Wynik Zmęczenia CNS: {report_bad_recovery['cns_fatigue_score']} / 100")
    print(f"  * Akcja systemu: {report_bad_recovery['status']}")
    print(f"  * Rekomendacja: {report_bad_recovery['coach_recommendation']}")
    print(f"  * Mnożnik na kolejny trening: {report_bad_recovery['next_session_volume_multiplier']}x")

    # Scenariusz 2: Mądry trening góry ciała, świetna technika, na masie (nadwyżka)
    smart_upper_session = [
        {"rpe": 8.0, "form_quality": 1.15, "is_compound": True},  # Idealne wiosłowanie z oparciem
        {"rpe": 8.5, "form_quality": 1.10, "is_compound": True},  # Stabilne wyciskanie
        {"rpe": 9.0, "form_quality": 1.05, "is_compound": False},  # Izolacja bocznego barku
        {"rpe": 9.0, "form_quality": 1.05, "is_compound": False}
    ]

    report_good_recovery = tracker.analyze_readiness(
        recent_sets=smart_upper_session,
        diet_modifier=1.20,  # Nadwyżka kaloryczna (Surplus)
        sleep_quality=1.10  # Świetny sen
    )

    print("\n[SCENARIUSZ B: Czysta Technika na Nadwyżce Kalorycznej]")
    print(f"  * Wynik Zmęczenia CNS: {report_good_recovery['cns_fatigue_score']} / 100")
    print(f"  * Akcja systemu: {report_good_recovery['status']}")
    print(f"  * Rekomendacja: {report_good_recovery['coach_recommendation']}")
    print(f"  * Mnożnik na kolejny trening: {report_good_recovery['next_session_volume_multiplier']}x")
    print("=" * 70)