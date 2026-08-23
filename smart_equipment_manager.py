"""
SMART EQUIPMENT MANAGER & SUBSTITUTOR (smart_equipment_manager.py)
------------------------------------------------------------------
Zarządza dostępnym sprzętem użytkownika. Filtruje bazę ćwiczeń i dobiera
najlepszy biomechaniczny zamiennik w czasie rzeczywistym, gdy docelowa
maszyna jest niedostępna lub zajęta.
"""

# --- 1. ZAAWANSOWANA BAZA ĆWICZEŃ (Z Tagami Sprzętowymi) ---

ENHANCED_DATABASE = {
    "Czworogłowe": [
        {
            "name": "Hack Przysiad na Maszynie",
            "equipment": {"hack_machine"},
            "biomech_score_func": lambda l: 10.0 if l["L_femur"] > 0.26 else 9.0
        },
        {
            "name": "Wypychanie na Suwnicy (Leg Press)",
            "equipment": {"leg_press_machine"},
            "biomech_score_func": lambda l: 9.5 if l["L_femur"] > 0.26 else 8.5
        },
        {
            "name": "Przysiad Bułgarski z Hantlami",
            "equipment": {"dumbbells", "bench"},
            "biomech_score_func": lambda l: 9.0  # Dobre niezależnie od dźwigni, ale wymaga stabilizacji
        },
        {
            "name": "Przysiad ze Sztangą (Back Squat)",
            "equipment": {"barbell", "squat_rack"},
            "biomech_score_func": lambda l: 5.0 if l["L_femur"] > 0.26 else 10.0  # Kara za długie uda
        }
    ],
    "Plecy": [
        {
            "name": "Wiosłowanie z Podparciem Klatki na Maszynie",
            "equipment": {"chest_supported_row_machine"},
            "biomech_score_func": lambda l: 10.0
        },
        {
            "name": "Wiosłowanie Hantlami w Oparciu o Ławkę Dodatnią",
            "equipment": {"dumbbells", "bench"},
            "biomech_score_func": lambda l: 9.5
        },
        {
            "name": "Wiosłowanie Półsztangą (T-Bar Row)",
            "equipment": {"barbell", "landmine_attachment"},
            "biomech_score_func": lambda l: 6.0 if l["L_torso"] > 0.27 else 9.0
        }
    ]
}


# --- 2. SILNIK ZAMIENNIKÓW ---

class EquipmentManager:
    def __init__(self, user_available_equipment: set, user_levers: dict):
        """
        user_available_equipment: Zbiór (set) stringów oznaczających sprzęt w obecnej siłowni.
        user_levers: Słownik proporcji kości (np. L_femur, L_torso) z profilu użytkownika.
        """
        self.available_equipment = set(user_available_equipment)
        self.levers = user_levers

    def add_equipment(self, item: str):
        self.available_equipment.add(item)

    def remove_equipment(self, item: str):
        if item in self.available_equipment:
            self.available_equipment.remove(item)

    def get_best_exercise(self, muscle_group: str, exclude_exercises: list = None):
        """
        Szuka najlepszego ćwiczenia dla danej partii, uwzględniając WYŁĄCZNIE
        dostępny sprzęt i sortując wyniki po punktacji biomechanicznej dźwigni.
        """
        if exclude_exercises is None:
            exclude_exercises = []

        available_exercises = ENHANCED_DATABASE.get(muscle_group, [])
        valid_options = []

        for ex in available_exercises:
            # Sprawdź, czy ćwiczenie nie zostało wykluczone (bo np. użytkownik zgłosił awarię)
            if ex["name"] in exclude_exercises:
                continue

            # Sprawdź, czy siłownia posiada wszystkie wymagane sprzęty do tego ćwiczenia
            # ex["equipment"].issubset(...) sprawdza czy zbiór wymagany mieści się w zbiorze dostępnym
            if ex["equipment"].issubset(self.available_equipment):
                score = ex["biomech_score_func"](self.levers)
                valid_options.append({"name": ex["name"], "score": score})

        if not valid_options:
            return {"error": f"Brak sprzętu na siłowni, aby przetrenować partię: {muscle_group}"}

        # Sortujemy od najlepszego do najgorszego (na podstawie dźwigni!)
        valid_options.sort(key=lambda x: x["score"], reverse=True)
        return valid_options[0]  # Zwraca top 1

    def swap_exercise_on_the_fly(self, muscle_group: str, currently_assigned: str):
        """
        Funkcja awaryjna. Użytkownik w aplikacji klika "Maszyna zajęta / Brak sprzętu".
        Algorytm szuka kolejnego najlepszego zastępstwa.
        """
        print(f"[ZAMIANA] Użytkownik zgłosił brak/zajętość: '{currently_assigned}'. Szukam zamiennika...")
        new_exercise = self.get_best_exercise(muscle_group, exclude_exercises=[currently_assigned])
        return new_exercise


# --- DEMONSTRACJA DZIAŁANIA W PRAKTYCE ---

if __name__ == "__main__":
    # Profil użytkownika z DŁUGIMI KOŚCIAMI UDOWYMI (Kara do tradycyjnego przysiadu)
    user_levers = {"L_femur": 0.29, "L_torso": 0.25}

    print("=" * 70)
    print("SCENARIUSZ 1: Użytkownik idzie na potężną siłownię komercyjną (Ma wszystko)")
    print("=" * 70)

    commercial_gym_equipment = {"hack_machine", "leg_press_machine", "dumbbells", "bench", "barbell", "squat_rack"}
    manager1 = EquipmentManager(commercial_gym_equipment, user_levers)

    best_quads = manager1.get_best_exercise("Czworogłowe")
    print(f"Wybrane ćwiczenie na Czworogłowe: {best_quads['name']} (Bio-Score: {best_quads['score']})")

    print("\n" + "=" * 70)
    print("SCENARIUSZ 2: Użytkownik jest w małej siłowni hotelowej (Tylko Hantle i Ławka)")
    print("=" * 70)

    hotel_gym_equipment = {"dumbbells", "bench"}
    manager2 = EquipmentManager(hotel_gym_equipment, user_levers)

    best_quads_hotel = manager2.get_best_exercise("Czworogłowe")
    print(f"Wybrane ćwiczenie na Czworogłowe: {best_quads_hotel['name']} (Bio-Score: {best_quads_hotel['score']})")

    print("\n" + "=" * 70)
    print("SCENARIUSZ 3: Siłownia ma sprzęt, ale maszyna jest ZAJĘTA w trakcie treningu")
    print("=" * 70)

    # Jesteśmy na dużej siłowni, algorytm zaplanował Hack Przysiad (jak w Scenariuszu 1).
    # Niestety 5 osób czeka w kolejce do maszyny.
    swapped_exercise = manager1.swap_exercise_on_the_fly("Czworogłowe", currently_assigned="Hack Przysiad na Maszynie")
    print(f"Nowe wybrane ćwiczenie: {swapped_exercise['name']} (Bio-Score: {swapped_exercise['score']})")
    print("=" * 70)