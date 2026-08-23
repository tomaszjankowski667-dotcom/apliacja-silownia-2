"""
SOCIAL, PROGRESSION & GAMIFICATION ENGINE (social_progress_engine.py)
---------------------------------------------------------------------
Zarządza dziennikiem treningowym w bazie SQLite.
Monitoruje adaptację, tworzy rankingi (Leaderboard 1-100 Bio-Score)
oraz rozstrzyga pojedynki 1v1 między znajomymi na podstawie biomechaniki.
"""

import sqlite3
from datetime import datetime, timedelta
import numpy as np


def init_db(db_name="fitness_biomechanics.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        gender TEXT,
        height_cm REAL,
        weight_kg REAL,
        lbm_kg REAL,
        l_femur REAL,
        l_torso REAL,
        l_humerus REAL,
        global_bio_score INTEGER,
        rank_title TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS workout_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp TEXT,
        exercise_name TEXT,
        muscle_group TEXT,
        weight_kg REAL,
        reps INTEGER,
        rpe REAL,
        form_quality REAL,
        session_hypertrophy_points REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()


class ProgressTracker:
    def __init__(self, db_name="fitness_biomechanics.db"):
        self.db_name = db_name

    def log_workout_set(self, user_id, exercise_name, muscle_group, weight_kg, reps, rpe, form_quality=1.0):
        effective_reps = max(1, reps - max(0, 10 - int(rpe)))
        set_points = round(weight_kg * effective_reps * form_quality * 0.1, 2)

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO workout_logs (user_id, timestamp, exercise_name, muscle_group, weight_kg, reps, rpe, form_quality, session_hypertrophy_points)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, datetime.now().isoformat(), exercise_name, muscle_group, weight_kg, reps, rpe, form_quality,
              set_points))
        conn.commit()
        conn.close()
        return set_points


class SocialArena:
    def __init__(self, db_name="fitness_biomechanics.db"):
        self.db_name = db_name

    def get_friends_leaderboard(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT name, global_bio_score, rank_title, lbm_kg FROM users ORDER BY global_bio_score DESC")
        results = cursor.fetchall()
        conn.close()

        leaderboard = [{"position": i + 1, "name": r[0], "bio_score": r[1], "rank_title": r[2], "lbm_kg": r[3]} for i, r
                       in enumerate(results)]
        return leaderboard

    def resolve_1v1_biomechanical_duel(self, user1_id, user2_id, exercise_category="quads", weight_u1=100,
                                       weight_u2=120):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT name, l_femur, l_humerus, l_torso, lbm_kg FROM users WHERE id IN (?, ?)",
                       (user1_id, user2_id))
        users = cursor.fetchall()
        conn.close()

        if len(users) < 2: return "Błąd: Brak danych."
        u1, u2 = users[0], users[1]

        lever_u1 = u1[1] if exercise_category == "quads" else (u1[2] if exercise_category == "chest" else u1[3])
        lever_u2 = u2[1] if exercise_category == "quads" else (u2[2] if exercise_category == "chest" else u2[3])

        G = 9.81
        score_u1 = round((weight_u1 * G * lever_u1) / u1[4], 2)
        score_u2 = round((weight_u2 * G * lever_u2) / u2[4], 2)

        winner = u1[0] if score_u1 > score_u2 else u2[0]
        return {"winner": winner, "score_1": score_u1, "score_2": score_u2}


if __name__ == "__main__":
    init_db()
    print("Baza danych i silnik postępów zainicjalizowane pomyślnie.")