"""
TEST JEDNOSTKOWY ANALIZY WIZYJNEJ (vision/test/test_vision_single.py)
--------------------------------------------------------------------
Uruchomienie z głównego katalogu:
python vision/test/test_vision_single.py --video seria.mp4 --exercise Flat_Dumbbell_Press
"""

import argparse
import os
import sys

# Dynamiczne dodanie głównego folderu projektu oraz podfolderu vision do sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
VISION_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for path in [PROJECT_ROOT, VISION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from vision_exercise_analyzer import analyze_video_with_model
except ImportError:
    try:
        from vision.vision_exercise_analyzer import analyze_video_with_model
    except ImportError as e:
        print(f"\n[BŁĄD IMPORTU] Nie znaleziono pliku vision_exercise_analyzer.py")
        print(f"Upewnij się, że plik vision_exercise_analyzer.py znajduje się w głównym folderze lub w folderze 'vision/'.")
        raise e

EXERCISES_LIST = [
    "Flat_Barbell_Press",
    "Flat_Dumbbell_Press",
    "Machine_Chest_Press",
    "Flat_Smith_Press",
    "Mid_Cable_Crossover",
    "Butterfly_Pec_Deck",
    "Flat_Dumbbell_Flyes",
    "Hex_Squeeze_Press"
]

def main():
    parser = argparse.ArgumentParser(description="Test analizatora wizyjnego wzorców 3D")
    parser.add_argument("--video", type=str, default="seria.mp4", help="Ścieżka do nagrania wideo")
    parser.add_argument("--exercise", type=str, default="Flat_Dumbbell_Press", choices=EXERCISES_LIST, help="Klucz ćwiczenia")
    parser.add_argument("--out", type=str, default="test_output.mp4", help="Plik wynikowy")

    args = parser.parse_args()

    # Jeśli podano ścieżkę względną, sprawdź zarówno w katalogu roboczym, jak i głównym
    video_path = args.video
    if not os.path.exists(video_path):
        alt_path = os.path.join(PROJECT_ROOT, args.video)
        if os.path.exists(alt_path):
            video_path = alt_path
        else:
            print(f"\n[BŁĄD] Nie znaleziono pliku wideo: '{args.video}'")
            print(f"Szukano w: {os.path.abspath(args.video)} oraz {alt_path}")
            return

    print(f"\n>>> Rozpoczynam test analizy wizyjnej dla: [{args.exercise}]")
    print(f">>> Plik wejściowy: {video_path}")
    print(f">>> Plik wyjściowy: {args.out}\n")

    analyze_video_with_model(video_path, args.out, exercise_key=args.exercise)
    print(f"\n[SUKCES] Wygenerowano wideo z naniesionym wzorcem 3D: {args.out}")

if __name__ == "__main__":
    main()