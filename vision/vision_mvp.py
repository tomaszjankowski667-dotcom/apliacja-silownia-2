import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque

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

traj_R = deque(maxlen=60)
traj_L = deque(maxlen=60)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = landmarker.detect(mp_image)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = image_bgr.shape

            r_sh = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            r_el = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            r_wr = (int(landmarks[16].x * w), int(landmarks[16].y * h))

            l_sh = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            l_el = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            l_wr = (int(landmarks[15].x * w), int(landmarks[15].y * h))

            traj_R.append(r_wr)
            traj_L.append(l_wr)

            cv2.line(image_bgr, r_sh, r_el, (255, 0, 0), 4)
            cv2.line(image_bgr, r_el, r_wr, (255, 0, 0), 4)
            cv2.line(image_bgr, l_sh, l_el, (255, 0, 0), 4)
            cv2.line(image_bgr, l_el, l_wr, (255, 0, 0), 4)
            cv2.line(image_bgr, r_sh, l_sh, (255, 0, 0), 4)

            cv2.circle(image_bgr, r_sh, 8, (0, 0, 255), -1)
            cv2.circle(image_bgr, r_el, 8, (0, 0, 255), -1)
            cv2.circle(image_bgr, r_wr, 10, (0, 255, 255), -1)
            cv2.circle(image_bgr, l_sh, 8, (0, 0, 255), -1)
            cv2.circle(image_bgr, l_el, 8, (0, 0, 255), -1)
            cv2.circle(image_bgr, l_wr, 10, (0, 255, 255), -1)

            for i in range(1, len(traj_R)):
                cv2.line(image_bgr, traj_R[i - 1], traj_R[i], (0, 165, 255), 3)
            for i in range(1, len(traj_L)):
                cv2.line(image_bgr, traj_L[i - 1], traj_L[i], (0, 165, 255), 3)

        cv2.imshow('ING Pitch - Vision MVP', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()