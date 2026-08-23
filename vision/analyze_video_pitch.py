import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import sys
from scipy.signal import find_peaks

# --- Pobranie modelu MediaPipe jeśli nie istnieje ---
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
    running_mode=VisionRunningMode.IMAGE)


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


def process_exercise_video(video_input_path, video_output_path="wynik_analizy.mp4"):
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (w, h))

    frames_rgb = []
    raw_l_wr, raw_l_el, raw_l_sh = [], [], []
    raw_r_wr, raw_r_el, raw_r_sh = [], [], []

    # 1. Detekcja punktów kluczowych MediaPipe
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

                sh_l = np.array([lm[11].x * w, lm[11].y * h])
                el_l = np.array([lm[13].x * w, lm[13].y * h])
                wr_l = np.array([lm[15].x * w, lm[15].y * h])

                sh_r = np.array([lm[12].x * w, lm[12].y * h])
                el_r = np.array([lm[14].x * w, lm[14].y * h])
                wr_r = np.array([lm[16].x * w, lm[16].y * h])

                raw_l_wr.append(wr_l)
                raw_l_el.append(el_l)
                raw_l_sh.append(sh_l)
                raw_r_wr.append(wr_r)
                raw_r_el.append(el_r)
                raw_r_sh.append(sh_r)
            else:
                last_l = raw_l_wr[-1] if raw_l_wr else np.array([0, 0])
                raw_l_wr.append(last_l)
                raw_l_el.append(raw_l_el[-1] if raw_l_el else np.array([0, 0]))
                raw_l_sh.append(raw_l_sh[-1] if raw_l_sh else np.array([0, 0]))
                raw_r_wr.append(raw_r_wr[-1] if raw_r_wr else np.array([0, 0]))
                raw_r_el.append(raw_r_el[-1] if raw_r_el else np.array([0, 0]))
                raw_r_sh.append(raw_r_sh[-1] if raw_r_sh else np.array([0, 0]))

    cap.release()
    total_frames = len(frames_rgb)
    if total_frames == 0:
        return

    # Wygładzanie trajektorii
    smooth_l_wr_pts = smooth_series(raw_l_wr, window_size=7)
    smooth_l_el_pts = smooth_series(raw_l_el, window_size=7)
    smooth_l_sh_pts = smooth_series(raw_l_sh, window_size=7)

    smooth_r_wr_pts = smooth_series(raw_r_wr, window_size=7)
    smooth_r_el_pts = smooth_series(raw_r_el, window_size=7)
    smooth_r_sh_pts = smooth_series(raw_r_sh, window_size=7)

    l_wr_y = np.array([p[1] for p in smooth_l_wr_pts])

    # 2. Analiza powtórzeń i precyzyjne wyznaczenie końca serii
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
    total_reps_detected = len(valid_rep_completion_frames)

    # Koniec aktywnego ćwiczenia przypada na zakończenie ostatniego powtórzenia (+ krótki bufor ~0.3s)
    if valid_rep_completion_frames:
        start_frame = 0
        end_frame = min(total_frames - 1, valid_rep_completion_frames[-1] + int(fps * 0.3))
    else:
        start_frame = 0
        end_frame = total_frames - 1

    active_y = l_wr_y[start_frame:end_frame + 1]
    active_rom_min = np.min(active_y) if len(active_y) > 0 else 0
    active_rom_max = np.max(active_y) if len(active_y) > 0 else 1
    rom_span = max(1.0, active_rom_max - active_rom_min)

    traj_left = []
    traj_right = []
    scores_history = []
    current_reps = 0
    final_score = None

    # 3. Rysowanie i zapis
    for f_idx in range(total_frames):
        frame = cv2.cvtColor(frames_rgb[f_idx], cv2.COLOR_RGB2BGR)
        is_active = (start_frame <= f_idx <= end_frame)

        l_sh = smooth_l_sh_pts[f_idx]
        l_el = smooth_l_el_pts[f_idx]
        l_wr = smooth_l_wr_pts[f_idx]

        r_sh = smooth_r_sh_pts[f_idx]
        r_el = smooth_r_el_pts[f_idx]
        r_wr = smooth_r_wr_pts[f_idx]

        # Rysowanie szkieletu
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

            t_val = np.clip((active_rom_max - l_wr[1]) / rom_span, 0.0, 1.0)
            target_x = int(l_sh[0] + (1.0 - t_val) * (l_el[0] - l_sh[0]) * 0.5)
            pos_dev = abs(l_wr[0] - target_x) / rom_span

            frame_score = int(np.clip(100 - (angle_dev * 40 + pos_dev * 30), 75, 98))
            scores_history.append(frame_score)

            cv2.circle(frame, (target_x, l_wr[1]), 8, (0, 255, 0), -1)

            # Panel na żywo
            cv2.rectangle(frame, (20, 20), (350, 130), (0, 0, 0), -1)
            cv2.putText(frame, f"Score: {frame_score}%", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {current_reps}", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Po zakończeniu ostatniego repa od razu wyświetla FINAL SCORE
            if scores_history:
                if final_score is None:
                    final_score = int(np.mean(scores_history))
                display_reps = total_reps_detected if total_reps_detected > 0 else current_reps
                cv2.rectangle(frame, (20, 20), (430, 135), (0, 0, 0), -1)
                cv2.putText(frame, f"FINAL SCORE: {final_score}%", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                            2)
                cv2.putText(frame, f"Total Reps: {display_reps} | Complete", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

        # Rysowanie linii trajektorii
        for i in range(1, len(traj_left)):
            cv2.line(frame, traj_left[i - 1], traj_left[i], (0, 165, 255), 3)
        for i in range(1, len(traj_right)):
            cv2.line(frame, traj_right[i - 1], traj_right[i], (0, 165, 255), 3)

        out.write(frame)

    out.release()
    print("Przetwarzanie zakończone!")


if __name__ == "__main__":
    process_exercise_video("test_exercise.mp4", "wynik_analizy.mp4")