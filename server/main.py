import os
import shutil
import subprocess
import urllib.request
from typing import List, Optional

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import openai
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.signal import find_peaks

app = FastAPI(title="Biomechanics AI Trainer API", version="1.0")

# Obsługa CORS pod aplikacje mobilne i przeglądarki
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Katalogi na pliki wejściowe i przetworzone nagrania
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Udostępnianie plików statycznych
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Inicjalizacja MediaPipe ---
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)


def smooth_series(data_list, window_size=7):
    if len(data_list) < window_size:
        return data_list
    arr = np.array(data_list, dtype=float)
    smoothed = np.copy(arr)
    pad = window_size // 2
    for d in range(arr.shape[1]):
        conv = np.convolve(arr[:, d], np.ones(window_size) / window_size, mode='same')
        smoothed[pad:-pad, d] = conv[pad:-pad]
    return [tuple(pt.astype(int)) for pt in smoothed]


def analyze_cv_core(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Nie można odczytać pliku wideo.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frames_rgb = []
    raw_l_wr, raw_l_el, raw_l_sh = [], [], []
    raw_r_wr, raw_r_el, raw_r_sh = [], [], []

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(img_rgb)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = landmarker.detect(mp_img)

            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                lm = res.pose_landmarks[0]
                raw_l_sh.append(np.array([lm[11].x * w, lm[11].y * h]))
                raw_l_el.append(np.array([lm[13].x * w, lm[13].y * h]))
                raw_l_wr.append(np.array([lm[15].x * w, lm[15].y * h]))

                raw_r_sh.append(np.array([lm[12].x * w, lm[12].y * h]))
                raw_r_el.append(np.array([lm[14].x * w, lm[14].y * h]))
                raw_r_wr.append(np.array([lm[16].x * w, lm[16].y * h]))
            else:
                raw_l_sh.append(raw_l_sh[-1] if raw_l_sh else np.array([0, 0]))
                raw_l_el.append(raw_l_el[-1] if raw_l_el else np.array([0, 0]))
                raw_l_wr.append(raw_l_wr[-1] if raw_l_wr else np.array([0, 0]))
                raw_r_sh.append(raw_r_sh[-1] if raw_r_sh else np.array([0, 0]))
                raw_r_el.append(raw_r_el[-1] if raw_r_el else np.array([0, 0]))
                raw_r_wr.append(raw_r_wr[-1] if raw_r_wr else np.array([0, 0]))

    cap.release()
    total_frames = len(frames_rgb)
    if total_frames == 0:
        raise Exception("Pusty plik wideo.")

    smooth_l_wr_pts = smooth_series(raw_l_wr, window_size=7)
    smooth_l_el_pts = smooth_series(raw_l_el, window_size=7)
    smooth_l_sh_pts = smooth_series(raw_l_sh, window_size=7)
    smooth_r_wr_pts = smooth_series(raw_r_wr, window_size=7)
    smooth_r_el_pts = smooth_series(raw_r_el, window_size=7)
    smooth_r_sh_pts = smooth_series(raw_r_sh, window_size=7)

    l_wr_y = np.array([p[1] for p in smooth_l_wr_pts])
    rom_span_val = np.max(l_wr_y) - np.min(l_wr_y)
    min_prom = rom_span_val * 0.30
    min_dist = int(fps * 0.8)
    bottom_peaks, _ = find_peaks(l_wr_y, prominence=min_prom, distance=min_dist)

    valid_rep_completion_frames = []
    top_threshold = np.min(l_wr_y) + rom_span_val * 0.40
    for peak in bottom_peaks:
        for f in range(peak, min(peak + int(fps * 2.5), total_frames)):
            if l_wr_y[f] <= top_threshold:
                valid_rep_completion_frames.append(f)
                break

    valid_rep_completion_frames = sorted(list(set(valid_rep_completion_frames)))
    total_reps = len(valid_rep_completion_frames)

    start_frame = 0
    end_frame = min(total_frames - 1, valid_rep_completion_frames[-1] + int(
        fps * 0.3)) if valid_rep_completion_frames else total_frames - 1

    active_y = l_wr_y[start_frame:end_frame + 1]
    active_rom_min = np.min(active_y) if len(active_y) > 0 else 0
    active_rom_max = np.max(active_y) if len(active_y) > 0 else 1
    rom_span = max(1.0, active_rom_max - active_rom_min)

    traj_left, traj_right = [], []
    scores_history = []
    angle_deviations = []
    current_reps = 0
    final_score = None

    for f_idx in range(total_frames):
        frame = cv2.cvtColor(frames_rgb[f_idx], cv2.COLOR_RGB2BGR)
        is_active = (start_frame <= f_idx <= end_frame)

        l_sh, l_el, l_wr = smooth_l_sh_pts[f_idx], smooth_l_el_pts[f_idx], smooth_l_wr_pts[f_idx]
        r_sh, r_el, r_wr = smooth_r_sh_pts[f_idx], smooth_r_el_pts[f_idx], smooth_r_wr_pts[f_idx]

        cv2.line(frame, l_sh, r_sh, (255, 0, 0), 4)
        cv2.line(frame, l_sh, l_el, (255, 0, 0), 4)
        cv2.line(frame, l_el, l_wr, (255, 0, 0), 4)
        cv2.line(frame, r_sh, r_el, (255, 0, 0), 4)
        cv2.line(frame, r_el, r_wr, (255, 0, 0), 4)

        for pt in [l_sh, l_el, l_wr, r_sh, r_el, r_wr]:
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)

        if is_active:
            traj_left.append(l_wr)
            traj_right.append(r_wr)

            if current_reps < len(valid_rep_completion_frames) and f_idx >= valid_rep_completion_frames[current_reps]:
                current_reps += 1

            forearm_vec = np.array([l_wr[0] - l_el[0], l_wr[1] - l_el[1]])
            forearm_len = np.linalg.norm(forearm_vec)
            angle_dev = abs(forearm_vec[0]) / forearm_len if forearm_len > 10 else 0.0
            angle_deviations.append(float(angle_dev))

            t_val = np.clip((active_rom_max - l_wr[1]) / rom_span, 0.0, 1.0)
            target_x = int(l_sh[0] + (1.0 - t_val) * (l_el[0] - l_sh[0]) * 0.5)
            pos_dev = abs(l_wr[0] - target_x) / rom_span

            frame_score = int(np.clip(100 - (angle_dev * 40 + pos_dev * 30), 75, 98))
            scores_history.append(frame_score)

            cv2.circle(frame, (target_x, l_wr[1]), 8, (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 20), (350, 130), (0, 0, 0), -1)
            cv2.putText(frame, f"Score: {frame_score}%", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {current_reps}", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            if scores_history:
                if final_score is None:
                    final_score = int(np.mean(scores_history))
                display_reps = total_reps if total_reps > 0 else current_reps
                cv2.rectangle(frame, (20, 20), (430, 135), (0, 0, 0), -1)
                cv2.putText(frame, f"FINAL SCORE: {final_score}%", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                            2)
                cv2.putText(frame, f"Total Reps: {display_reps} | Complete", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

        for i in range(1, len(traj_left)):
            cv2.line(frame, traj_left[i - 1], traj_left[i], (0, 165, 255), 3)
        for i in range(1, len(traj_right)):
            cv2.line(frame, traj_right[i - 1], traj_right[i], (0, 165, 255), 3)

        out.write(frame)

    out.release()

    # Automatyczna konwersja do formatu H.264 kompatybilnego z przeglądarkami smartfonów
    temp_output = output_path.replace(".mp4", "_temp.mp4")
    if os.path.exists(output_path):
        os.replace(output_path, temp_output)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe, "-y", "-i", temp_output,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_output):
            os.remove(temp_output)

    avg_score = int(np.mean(scores_history)) if scores_history else 0
    avg_angle_deg = round(float(np.mean(angle_deviations)) * 57.3, 1) if angle_deviations else 0.0

    return {
        "score": avg_score,
        "reps": total_reps,
        "avg_forearm_angle_dev_deg": avg_angle_deg,
        "rom_pixels": int(rom_span)
    }


def get_ai_coaching_cue(exercise: str, score: int, reps: int, angle_dev: float) -> str:
    """Generuje feedback trenerski za pomocą OpenAI lub silnika regułowego."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            prompt = f"""
            Jesteś elitarnym trenerem przygotowania motorycznego i biomechaniki.
            Użytkownik właśnie wykonał serię ćwiczenia: {exercise}.
            Parametry serii z systemu Computer Vision:
            - Wynik zgodności techniki: {score}%
            - Liczba poprawnych powtórzeń: {reps}
            - Średnie odchylenie pionu przedramion: {angle_dev} stopni.

            Napisz krótki, konkretny feedback (dokładnie 2 zwięzłe zdania):
            1. Pochwal element wykonany najlepiej.
            2. Podaj jedną praktyczną wskazówkę (tzw. cueing motoryczny) na kolejną serię.
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    # Fallback regułowy (gdy brak klucza API lub brak internetu)
    if score >= 90:
        return f"Świetna seria! Zachowałeś niemal idealny tor ruchu i stabilność przedramion ({reps} powt.). W kolejnej serii skup się na kontrolowaniu 2-sekundowej fazy opuszczania ciężaru."
    elif score >= 75:
        return f"Dobra technika bazowa ({score}%), ale na dole ruchu przedramię lekko ucieka od pionu ({angle_dev}°). Wkręć łopatki w ławkę i prowadź łokcie nieco bliżej tułowia."
    else:
        return f"Seria zaliczona ({reps} powt.), ale zwróć uwagę na tor ruchu. Skup się na prowadzeniu hantli po łuku nad klatkę i unikaj rozszerzania łokci na boki."


# --- Modele danych dla wektorów Edge ---

class LandmarkFrame(BaseModel):
    frame_idx: int
    l_sh: List[float]
    l_el: List[float]
    l_wr: List[float]
    r_sh: List[float]
    r_el: List[float]
    r_wr: List[float]


class PosePayload(BaseModel):
    exercise: str = "dumbbell_bench_press"
    landmarks: List[LandmarkFrame]


# --- Endpointy API ---

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/api/analyze")
async def analyze_video_endpoint(
        file: UploadFile = File(...),
        exercise: str = Form("dumbbell_bench_press")
):
    input_path = os.path.join(UPLOAD_DIR, f"raw_{file.filename}")
    output_filename = f"analyzed_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        metrics = analyze_cv_core(input_path, output_path)
        ai_cue = get_ai_coaching_cue(
            exercise=exercise,
            score=metrics["score"],
            reps=metrics["reps"],
            angle_dev=metrics["avg_forearm_angle_dev_deg"]
        )

        return {
            "success": True,
            "exercise": exercise,
            "score": metrics["score"],
            "reps": metrics["reps"],
            "metrics": {
                "forearm_deviation_deg": metrics["avg_forearm_angle_dev_deg"],
                "rom_px": metrics["rom_pixels"]
            },
            "ai_coach_feedback": ai_cue,
            "processed_video_url": f"/static/outputs/{output_filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-landmarks")
async def analyze_landmarks_endpoint(payload: PosePayload):
    frames = payload.landmarks
    if not frames:
        raise HTTPException(status_code=400, detail="Brak danych wektorowych.")

    l_wr_y = np.array([f.l_wr[1] for f in frames])
    rom_span = float(np.max(l_wr_y) - np.min(l_wr_y)) if len(l_wr_y) > 0 else 1.0

    min_prom = rom_span * 0.30
    peaks, _ = find_peaks(l_wr_y, prominence=min_prom, distance=15)
    total_reps = len(peaks)

    angle_deviations = []
    scores = []
    for f in frames:
        vec = np.array([f.l_wr[0] - f.l_el[0], f.l_wr[1] - f.l_el[1]])
        norm = np.linalg.norm(vec)
        dev = abs(vec[0]) / norm if norm > 10 else 0.0
        angle_deviations.append(float(dev))
        scores.append(int(np.clip(100 - dev * 40, 70, 98)))

    avg_score = int(np.mean(scores)) if scores else 85
    avg_angle_deg = round(float(np.mean(angle_deviations)) * 57.3, 1) if angle_deviations else 0.0

    ai_cue = get_ai_coaching_cue(
        exercise=payload.exercise,
        score=avg_score,
        reps=total_reps,
        angle_dev=avg_angle_deg
    )

    return {
        "success": True,
        "score": avg_score,
        "reps": total_reps,
        "metrics": {
            "forearm_deviation_deg": avg_angle_deg,
            "rom_px": int(rom_span)
        },
        "ai_coach_feedback": ai_cue
    }