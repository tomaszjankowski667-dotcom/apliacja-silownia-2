"""
Universal, layered exercise-video analyzer.

The module is deliberately conservative: it does not turn missing pose data into
invented measurements or render reconstructed anatomy as if it were observed.
Only observations that pass the independent anatomical and tracking quality
gates are drawn or scored.

Usage:
    python vision/vision_exercise_analyzer.py --video seria.mp4 --out wynik.mp4 \
        --exercise Flat_Barbell_Press
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.runtime_config import configure_deterministic_runtime

RUNTIME_CONFIG = configure_deterministic_runtime()

import cv2
import mediapipe as mp
import numpy as np

from vision.exercise_compatibility import (
    ExerciseMismatchError,
    evaluate_barbell_press_preflight,
    select_requested_exercise_track,
)
from vision.exercise_registry import ExerciseSpec, get_exercise_spec

cv2.setNumThreads(RUNTIME_CONFIG.opencv_threads)

DEFAULT_MODEL_NAME = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
NO_COMPLETE_REPS_MESSAGE = (
    "The equipment tracker did not find any complete repetitions. "
    "A complete repetition needs a visible top-bottom-top bar path."
)

# MediaPipe landmark indices.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24


@dataclass
class PoseCandidate:
    """One detected person in one frame, represented in image pixels."""

    frame_index: int
    points: dict[int, np.ndarray]
    visibility: dict[int, float]
    center: np.ndarray
    lower_y: float
    body_scale: float


@dataclass
class TrackSample:
    frame_index: int
    candidate: PoseCandidate


@dataclass
class PersonTrack:
    """A lightweight identity track built from stable body-centre continuity."""

    track_id: int
    samples: list[TrackSample] = field(default_factory=list)
    last_center: Optional[np.ndarray] = None
    last_scale: float = 1.0
    last_frame: int = -1

    def append(self, candidate: PoseCandidate) -> None:
        self.samples.append(TrackSample(candidate.frame_index, candidate))
        self.last_center = candidate.center
        self.last_scale = candidate.body_scale
        self.last_frame = candidate.frame_index


@dataclass
class PerspectiveCalibration:
    scale_px_per_m: float
    yaw_radians: float
    shoulder_width_px: float
    arm_length_px: float
    facing_direction: int


@dataclass
class Rep:
    start_frame: int
    bottom_frame: int
    end_frame: int
    rom_px: float
    score: Optional[int] = None


@dataclass
class FrameReference:
    """Immutable analysis origin acquired from the first input frame."""

    frame_index: int
    bar_center: np.ndarray
    plate_radius: float


@dataclass
class ScoreBreakdown:
    """Independent, quality-gated scores for equipment and limb movement."""

    bar_rep_scores: dict[int, int] = field(default_factory=dict)
    limb_rep_scores: dict[int, int] = field(default_factory=dict)
    overall_rep_scores: dict[int, int] = field(default_factory=dict)
    bar_frame_scores: dict[int, float] = field(default_factory=dict)
    limb_frame_scores: dict[int, float] = field(default_factory=dict)


def _aligned_rep_score_rows(breakdown: ScoreBreakdown) -> list[dict[str, Optional[int]]]:
    """Keep component scores attached to the same original detected repetition."""
    return [
        {
            "detected_rep_number": rep_index + 1,
            "bar_path_score": breakdown.bar_rep_scores[rep_index],
            "limb_motion_score": breakdown.limb_rep_scores.get(rep_index),
            "final_score": breakdown.overall_rep_scores.get(rep_index),
        }
        for rep_index in sorted(breakdown.bar_rep_scores)
    ]


def _point(landmark: object, width: int, height: int) -> np.ndarray:
    return np.array([float(landmark.x) * width, float(landmark.y) * height], dtype=float)


def _visibility(landmark: object) -> float:
    """MediaPipe task landmarks may omit visibility in older model versions."""
    value = getattr(landmark, "visibility", 1.0)
    return float(value) if value is not None else 1.0


def _valid_point(point: Optional[np.ndarray]) -> bool:
    return point is not None and bool(np.all(np.isfinite(point)))


def _median_point(points: Iterable[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = [point for point in points if _valid_point(point)]
    return np.median(np.asarray(valid, dtype=float), axis=0) if valid else None


def _clip_rect(x: int, y: int, width: int, height: int, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(0, min(width, frame_width - x))
    height = max(0, min(height, frame_height - y))
    return x, y, width, height


def _ensure_model(model_path: Optional[str | Path] = None) -> Path:
    """Return a local pose model, downloading the public MediaPipe model once."""
    destination = Path(model_path) if model_path else PROJECT_ROOT / DEFAULT_MODEL_NAME
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        print("Downloading MediaPipe pose model (one-time setup)...")
        urllib.request.urlretrieve(MODEL_URL, temporary)
        temporary.replace(destination)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "The MediaPipe pose model is missing and could not be downloaded. "
            "Connect the workspace to the internet once or place "
            f"'{DEFAULT_MODEL_NAME}' in the project root."
        ) from error
    return destination


def _load_exercise_model(
    exercise_key: str,
) -> tuple[ExerciseSpec, dict, dict, Callable]:
    """Load lever lengths and the existing 3D motion template without globals."""
    spec = get_exercise_spec(exercise_key)

    try:
        from user_data import PROFILES
        template_module = importlib.import_module(spec.template_module)
    except ImportError as error:
        raise RuntimeError(
            "Could not load the local exercise templates. "
            f"The analyzer requires user_data.py and {spec.template_module}."
        ) from error

    profile = next(iter(PROFILES.values()), {}) if isinstance(PROFILES, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    levers = dict(profile.get("levers", {}))
    levers.setdefault("L_humerus", 0.326)
    levers.setdefault("L_forearm", 0.285)
    levers.setdefault("biacromial_width", 0.41)
    levers.setdefault("L_torso", 0.50)
    levers.setdefault("chest_block", 0.24)

    database = template_module.get_exercises_data(profile)
    if exercise_key not in database:
        available = ", ".join(sorted(database))
        raise ValueError(f"Unknown exercise '{exercise_key}'. Available: {available}")
    exercise = database[exercise_key]
    return spec, levers, exercise, exercise["trajectory_func"]


# ---------------------------------------------------------------------------
# LAYER 1 — SPATIO-TEMPORAL DRY RUN
# ---------------------------------------------------------------------------

def _candidate_from_landmarks(landmarks: list[object], frame_index: int, width: int, height: int) -> Optional[PoseCandidate]:
    required = (
        LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    if any(index >= len(landmarks) for index in required):
        return None

    points = {index: _point(landmarks[index], width, height) for index in range(min(len(landmarks), 33))}
    visibility = {index: _visibility(landmarks[index]) for index in points}
    shoulders = [points[LEFT_SHOULDER], points[RIGHT_SHOULDER]]
    hips = [points[LEFT_HIP], points[RIGHT_HIP]]
    center = np.mean(np.asarray(shoulders + hips), axis=0)
    shoulder_width = float(np.linalg.norm(points[LEFT_SHOULDER] - points[RIGHT_SHOULDER]))
    torso_lengths = [
        np.linalg.norm(points[LEFT_SHOULDER] - points[LEFT_HIP]),
        np.linalg.norm(points[RIGHT_SHOULDER] - points[RIGHT_HIP]),
    ]
    body_scale = max(24.0, shoulder_width, float(np.median(torso_lengths)))
    # Use the torso and knees to describe where the person is in frame. Feet
    # are deliberately excluded: a standing assistant can extend one foot low
    # into the image without being the person doing the exercise.
    lower_body_indices = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, 25, 26)
    lower_y = max(float(points[index][1]) for index in lower_body_indices if index in points)
    return PoseCandidate(frame_index, points, visibility, center, lower_y, body_scale)


def _detect_candidates(
    frames_bgr: list[np.ndarray],
    model_path: Path,
    min_confidence: float,
) -> list[list[PoseCandidate]]:
    """Detect all people in every frame; identity is resolved in a separate pass."""
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=6,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    all_candidates: list[list[PoseCandidate]] = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        for frame_index, frame in enumerate(frames_bgr):
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(image)
            candidates = [
                candidate
                for landmarks in result.pose_landmarks
                if (candidate := _candidate_from_landmarks(landmarks, frame_index, width, height)) is not None
            ]
            all_candidates.append(candidates)
    return all_candidates


def _build_person_tracks(candidates_by_frame: list[list[PoseCandidate]], fps: float) -> list[PersonTrack]:
    """Associate persons by torso continuity, never by detector output order."""
    tracks: list[PersonTrack] = []
    next_track_id = 1
    max_gap = max(3, int(round(fps * 0.8)))

    for frame_index, candidates in enumerate(candidates_by_frame):
        active = [track for track in tracks if frame_index - track.last_frame <= max_gap]
        possible: list[tuple[float, int, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            for track_index, track in enumerate(active):
                if track.last_center is None:
                    continue
                distance = float(np.linalg.norm(candidate.center - track.last_center))
                scale_ratio = max(candidate.body_scale, track.last_scale) / max(1.0, min(candidate.body_scale, track.last_scale))
                gate = max(80.0, 1.9 * max(candidate.body_scale, track.last_scale))
                if distance <= gate and scale_ratio < 2.25:
                    possible.append((distance / max(1.0, candidate.body_scale), candidate_index, track_index))

        used_candidates: set[int] = set()
        used_tracks: set[int] = set()
        for _, candidate_index, track_index in sorted(possible):
            if candidate_index in used_candidates or track_index in used_tracks:
                continue
            active[track_index].append(candidates[candidate_index])
            used_candidates.add(candidate_index)
            used_tracks.add(track_index)

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index not in used_candidates:
                track = PersonTrack(track_id=next_track_id)
                next_track_id += 1
                track.append(candidate)
                tracks.append(track)
    return tracks


def _smooth_1d(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    window = min(window if window % 2 else window - 1, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values.copy()
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _turning_points(values: np.ndarray, minimum_distance: int, minimum_prominence: float, maxima: bool) -> list[int]:
    """Small dependency-free peak finder used for both dry run and repetition detection."""
    if len(values) < 3:
        return []
    # Evaluate prominence over a meaningful fraction of a repetition rather
    # than only a few adjacent frames of a smooth movement.
    radius = max(2, minimum_distance)
    result: list[int] = []
    for index in range(radius, len(values) - radius):
        value = values[index]
        left = values[index - radius:index]
        right = values[index + 1:index + 1 + radius]
        is_extreme = value >= np.max(left) and value >= np.max(right) if maxima else value <= np.min(left) and value <= np.min(right)
        local_floor = min(float(np.min(left)), float(np.min(right)))
        local_ceiling = max(float(np.max(left)), float(np.max(right)))
        prominence = value - local_floor if maxima else local_ceiling - value
        if not is_extreme or prominence < minimum_prominence:
            continue
        if result and index - result[-1] < minimum_distance:
            previous = result[-1]
            is_better = value > values[previous] if maxima else value < values[previous]
            if is_better:
                result[-1] = index
        else:
            result.append(index)
    return result


def _track_motion_score(track: PersonTrack, frame_height: int, fps: float) -> Optional[float]:
    """Score low placement plus repeat-like vertical motion, suppressing stationary spotters."""
    if len(track.samples) < max(10, int(fps * 0.7)):
        return None
    series_length = track.last_frame - track.samples[0].frame_index + 1
    wrist_y = np.full(series_length, np.nan, dtype=float)
    lower_values: list[float] = []
    scales: list[float] = []
    for sample in track.samples:
        candidate = sample.candidate
        wrists = [
            candidate.points.get(index)
            for index in (LEFT_WRIST, RIGHT_WRIST)
            if candidate.visibility.get(index, 0.0) >= 0.35
        ]
        wrists = [point for point in wrists if _valid_point(point)]
        if not wrists:
            continue
        wrist_y[sample.frame_index - track.samples[0].frame_index] = float(np.mean([point[1] for point in wrists]))
        lower_values.append(candidate.lower_y)
        scales.append(candidate.body_scale)
    valid_indices = np.flatnonzero(np.isfinite(wrist_y))
    if len(valid_indices) < max(8, int(fps * 0.5)):
        return None

    # Interpolate only short detector dropouts; do not let distant observations
    # become adjacent samples in a fabricated movement cycle.
    for index in range(len(wrist_y)):
        if math.isfinite(float(wrist_y[index])):
            continue
        left, right = index - 1, index + 1
        while left >= 0 and not math.isfinite(float(wrist_y[left])):
            left -= 1
        while right < len(wrist_y) and not math.isfinite(float(wrist_y[right])):
            right += 1
        if left >= 0 and right < len(wrist_y) and right - left <= max(3, int(fps * 0.25)):
            wrist_y[index] = wrist_y[left] + (wrist_y[right] - wrist_y[left]) * ((index - left) / (right - left))
    valid_values = wrist_y[np.isfinite(wrist_y)]
    motion = float(np.percentile(valid_values, 90) - np.percentile(valid_values, 10))
    scale = max(1.0, float(np.median(scales)))
    normalized_motion = min(1.0, motion / (scale * 0.65))
    maxima: list[int] = []
    minima: list[int] = []
    start = 0
    while start < len(wrist_y):
        while start < len(wrist_y) and not math.isfinite(float(wrist_y[start])):
            start += 1
        end = start
        while end < len(wrist_y) and math.isfinite(float(wrist_y[end])):
            end += 1
        if end - start >= max(6, int(fps * 0.35)):
            smooth = _smooth_1d(wrist_y[start:end])
            maxima.extend(
                start + index
                for index in _turning_points(smooth, max(2, int(fps * 0.28)), max(2.0, motion * 0.12), maxima=True)
            )
            minima.extend(
                start + index
                for index in _turning_points(smooth, max(2, int(fps * 0.28)), max(2.0, motion * 0.12), maxima=False)
            )
        start = end
    # A low person standing still must never be eligible merely because of
    # their image position; it needs a material vertical excursion and a turn.
    if normalized_motion < 0.22 or not maxima or not minima:
        return None
    cyclicity = min(1.0, min(len(maxima), len(minima)) / 2.0)
    low_in_frame = min(1.0, float(np.median(lower_values)) / max(1.0, frame_height))
    return 0.35 * low_in_frame + 0.50 * normalized_motion + 0.15 * cyclicity


def _choose_lifter(tracks: list[PersonTrack], frame_height: int, fps: float) -> PersonTrack:
    """Select the low, cyclically moving person rather than an assistant in the background."""
    scored = [(score, track) for track in tracks if (score := _track_motion_score(track, frame_height, fps)) is not None]
    if not scored:
        raise RuntimeError(
            "No continuous lifter track with sufficient vertical movement was found. "
            "Use a clip where the exerciser and at least one arm are visible for several seconds."
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.045:
        raise RuntimeError(
            "Two people have similarly strong movement tracks, so the lifter cannot be selected reliably. "
            "Crop the clip to the exerciser or use a less crowded camera angle."
        )
    return scored[0][1]


def _choose_near_arm(track: PersonTrack, requested_side: Optional[str] = None) -> str:
    """Use screen-Y depth cue; a right-side profile normally resolves to the right arm."""
    if requested_side in {"left", "right"}:
        return requested_side
    depth_evidence: list[tuple[float, float]] = []
    for sample in track.samples:
        candidate = sample.candidate
        points = candidate.points
        right = [points.get(RIGHT_SHOULDER), points.get(RIGHT_ELBOW), points.get(RIGHT_WRIST)]
        left = [points.get(LEFT_SHOULDER), points.get(LEFT_ELBOW), points.get(LEFT_WRIST)]
        required_indices = (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        if all(candidate.visibility.get(index, 0.0) >= 0.35 for index in required_indices) and all(_valid_point(point) for point in right + left):
            difference = float(np.mean([point[1] for point in right]) - np.mean([point[1] for point in left]))
            depth_evidence.append((difference, candidate.body_scale))
    # In an image, the camera-near side typically projects lower (larger screen Y).
    # This makes a conventional profile from the right choose the right arm.
    if len(depth_evidence) < 8:
        raise RuntimeError(
            "The camera-near arm cannot be determined from enough visible landmark observations. "
            "Pass near_arm='left' or near_arm='right' only if the recording orientation is known."
        )
    differences = np.asarray([item[0] for item in depth_evidence], dtype=float)
    scale = max(1.0, float(np.median([item[1] for item in depth_evidence])))
    median_difference = float(np.median(differences))
    sign_consistency = float(np.mean(np.sign(differences) == np.sign(median_difference)))
    if abs(median_difference) < scale * 0.045 or sign_consistency < 0.72:
        raise RuntimeError(
            "The camera-near arm is visually ambiguous. "
            "Use a clearer profile angle or pass a verified near_arm override."
        )
    return "right" if median_difference >= 0 else "left"


def _paths_from_track(track: PersonTrack, near_arm: str, frame_count: int) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]], list[Optional[np.ndarray]], list[Optional[np.ndarray]], list[float]]:
    shoulder_index, elbow_index, wrist_index, hip_index = (
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP)
        if near_arm == "right"
        else (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP)
    )
    shoulders: list[Optional[np.ndarray]] = [None] * frame_count
    elbows: list[Optional[np.ndarray]] = [None] * frame_count
    wrists: list[Optional[np.ndarray]] = [None] * frame_count
    hips: list[Optional[np.ndarray]] = [None] * frame_count
    quality: list[float] = [0.0] * frame_count
    for sample in track.samples:
        candidate = sample.candidate
        index = sample.frame_index
        shoulders[index] = candidate.points.get(shoulder_index)
        elbows[index] = candidate.points.get(elbow_index)
        wrists[index] = candidate.points.get(wrist_index)
        hips[index] = candidate.points.get(hip_index)
        quality[index] = min(
            candidate.visibility.get(shoulder_index, 0.0),
            candidate.visibility.get(elbow_index, 0.0),
            candidate.visibility.get(wrist_index, 0.0),
        )
    return shoulders, elbows, wrists, hips, quality


# ---------------------------------------------------------------------------
# LAYER 2 — HARD ANATOMY AND INVERSE KINEMATICS
# ---------------------------------------------------------------------------

def _solve_elbow_ik(shoulder: np.ndarray, wrist: np.ndarray, humerus_px: float, forearm_px: float, expected_elbow: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Return the anatomically reachable elbow solution nearest to the observed side."""
    displacement = wrist - shoulder
    distance = float(np.linalg.norm(displacement))
    minimum_reach = abs(humerus_px - forearm_px) + 1e-3
    maximum_reach = humerus_px + forearm_px - 1e-3
    if not minimum_reach <= distance <= maximum_reach:
        return None
    base = math.atan2(displacement[1], displacement[0])
    cosine = np.clip(
        (humerus_px**2 + distance**2 - forearm_px**2) / (2.0 * humerus_px * distance),
        -1.0,
        1.0,
    )
    angle = math.acos(float(cosine))
    options = [
        shoulder + humerus_px * np.array([math.cos(base + angle), math.sin(base + angle)]),
        shoulder + humerus_px * np.array([math.cos(base - angle), math.sin(base - angle)]),
    ]
    if expected_elbow is not None and _valid_point(expected_elbow):
        return min(options, key=lambda option: float(np.linalg.norm(option - expected_elbow)))
    # For bench footage the elbow is usually lower than the locked shoulder.
    return max(options, key=lambda option: float(option[1]))


def _calibrate_perspective(
    shoulders: list[Optional[np.ndarray]],
    hips: list[Optional[np.ndarray]],
    track: PersonTrack,
    levers: dict,
    landmark_quality: list[float],
) -> PerspectiveCalibration:
    near_shoulder = _median_point(
        shoulder for shoulder, quality in zip(shoulders, landmark_quality) if quality >= 0.35
    )
    near_hip = _median_point(hips)
    if near_shoulder is None or near_hip is None:
        raise RuntimeError("The selected lifter has no stable shoulder and hip anchor.")

    shoulder_pairs: list[float] = []
    upper_arm_observations: list[tuple[float, np.ndarray]] = []
    for sample in track.samples:
        points = sample.candidate.points
        left_shoulder, right_shoulder = points.get(LEFT_SHOULDER), points.get(RIGHT_SHOULDER)
        if (
            sample.candidate.visibility.get(LEFT_SHOULDER, 0.0) >= 0.35
            and sample.candidate.visibility.get(RIGHT_SHOULDER, 0.0) >= 0.35
            and _valid_point(left_shoulder)
            and _valid_point(right_shoulder)
        ):
            shoulder_pairs.append(float(np.linalg.norm(left_shoulder - right_shoulder)))
        for shoulder_index, elbow_index in ((LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)):
            shoulder, elbow = points.get(shoulder_index), points.get(elbow_index)
            visible = (
                sample.candidate.visibility.get(shoulder_index, 0.0) >= 0.35
                and sample.candidate.visibility.get(elbow_index, 0.0) >= 0.35
            )
            # Reject the same impossible "elbow above shoulder" geometry used
            # later by the scoring gate before it can affect projection scale.
            if visible and _valid_point(shoulder) and _valid_point(elbow):
                upper_arm_observations.append((float(np.linalg.norm(elbow - shoulder)), shoulder))

    if len(shoulder_pairs) < 5 or len(upper_arm_observations) < 5:
        raise RuntimeError("There are not enough visible shoulder/arm measurements to calibrate camera perspective.")
    shoulder_width_px = float(np.median(shoulder_pairs))
    raw_arm_median = float(np.median([length for length, _ in upper_arm_observations]))
    upper_arm_lengths = [
        length
        for length, shoulder in upper_arm_observations
        if raw_arm_median * 0.55 <= length <= raw_arm_median * 1.45
        and float(np.linalg.norm(shoulder - near_shoulder)) <= max(shoulder_width_px * 1.5, raw_arm_median * 1.5)
    ]
    if len(upper_arm_lengths) < 5:
        raise RuntimeError("Arm measurements are inconsistent or occluded; perspective calibration was rejected.")
    arm_length_px = float(np.median(upper_arm_lengths))
    humerus_m = float(levers["L_humerus"])
    shoulder_width_m = float(levers["biacromial_width"])
    scale_from_arm = arm_length_px / max(humerus_m, 0.05)
    expected_shoulder_width = scale_from_arm * shoulder_width_m
    shoulder_ratio = np.clip(shoulder_width_px / max(expected_shoulder_width, 1.0), 0.08, 1.0)
    yaw_radians = float(math.acos(float(shoulder_ratio)))
    facing_direction = -1 if near_hip[0] > near_shoulder[0] else 1
    return PerspectiveCalibration(scale_from_arm, yaw_radians, shoulder_width_px, arm_length_px, facing_direction)


def _initial_pose_anchor(
    shoulders: list[Optional[np.ndarray]],
    elbows: list[Optional[np.ndarray]],
    wrists: list[Optional[np.ndarray]],
    hips: list[Optional[np.ndarray]],
    landmark_quality: list[float],
    fps: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Use only the first short, trustworthy pose window as the body reference."""
    valid_indices = [
        index
        for index, (shoulder, elbow, wrist, hip, quality) in enumerate(
            zip(shoulders, elbows, wrists, hips, landmark_quality)
        )
        if (
            quality >= 0.35
            and _valid_point(shoulder)
            and _valid_point(elbow)
            and _valid_point(wrist)
            and _valid_point(hip)
        )
    ]
    if not valid_indices:
        raise RuntimeError("No trustworthy shoulder/hip observation was found for the selected lifter.")
    first_index = valid_indices[0]
    window_end = first_index + max(5, int(round(fps * 0.50)))
    window = [index for index in valid_indices if index < window_end]
    if len(window) < 3:
        raise RuntimeError("The initial body pose is too brief or occluded to establish an anatomical reference.")
    shoulder = _median_point(shoulders[index] for index in window)
    hip = _median_point(hips[index] for index in window)
    if shoulder is None or hip is None:
        raise RuntimeError("Could not establish the initial shoulder/bench anchor.")
    return first_index, shoulder, hip


def _valid_observed_elbow(
    shoulder: Optional[np.ndarray],
    elbow: Optional[np.ndarray],
    humerus_px: float,
    quality: float,
) -> bool:
    if not (_valid_point(shoulder) and _valid_point(elbow)):
        return False
    distance = float(np.linalg.norm(elbow - shoulder))
    return quality >= 0.35 and humerus_px * 0.45 <= distance <= humerus_px * 1.45


def _valid_arm_chain(
    shoulder: Optional[np.ndarray],
    elbow: Optional[np.ndarray],
    wrist: Optional[np.ndarray],
    humerus_px: float,
    forearm_px: float,
    quality: float,
) -> bool:
    """Validate an observed 2-D arm without crashing on frontal perspectives."""
    if (
        quality < 0.35
        or not _valid_point(shoulder)
        or not _valid_point(elbow)
        or not _valid_point(wrist)
    ):
        return False
    observed_humerus = float(np.linalg.norm(elbow - shoulder))
    observed_forearm = float(np.linalg.norm(wrist - elbow))
    observed_reach = float(np.linalg.norm(wrist - shoulder))
    return (
        humerus_px * 0.45 <= observed_humerus <= humerus_px * 1.45
        and forearm_px * 0.45 <= observed_forearm <= forearm_px * 1.45
        and observed_reach <= (humerus_px + forearm_px) * 1.08
    )


def _anatomy_reliability_warning(
    shoulders: list[Optional[np.ndarray]],
    elbows: list[Optional[np.ndarray]],
    wrists: list[Optional[np.ndarray]],
    hips: list[Optional[np.ndarray]],
    landmark_quality: list[float],
    anchor_shoulder: Optional[np.ndarray],
    humerus_px: float,
    forearm_px: float,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> Optional[str]:
    """Return a reason to suppress anatomy when joint evidence is insufficient."""
    if not _valid_point(anchor_shoulder) or end_frame < start_frame:
        return "No stable frame-one shoulder anchor was available for an anatomical score."
    if not (
        _valid_point(shoulders[start_frame])
        and _valid_point(elbows[start_frame])
        and _valid_point(wrists[start_frame])
        and _valid_point(hips[start_frame])
        and _valid_arm_chain(
            shoulders[start_frame],
            elbows[start_frame],
            wrists[start_frame],
            humerus_px,
            forearm_px,
            landmark_quality[start_frame],
        )
    ):
        return "Frame 1 does not contain a complete, geometrically valid arm and hip observation."

    indices = list(range(max(0, start_frame), min(len(landmark_quality), end_frame + 1)))
    if len(indices) < max(6, int(round(fps * 0.35))):
        return "The analyzed movement segment is too short to validate joint anatomy."

    visible_indices = [
        index
        for index in indices
        if (
            landmark_quality[index] >= 0.15
            and _valid_point(shoulders[index])
            and _valid_point(elbows[index])
            and _valid_point(wrists[index])
            and _valid_point(hips[index])
        )
    ]
    visibility_coverage = len(visible_indices) / len(indices)
    if visibility_coverage < 0.25:
        return "Joint visibility is too fragmented across the analyzed set for a full anatomical score."

    qualities = np.asarray([landmark_quality[index] for index in visible_indices], dtype=float)
    if float(np.median(qualities)) < 0.3:
        return "Joint confidence is too low for a full anatomical score."

    valid_indices = [
        index
        for index in visible_indices
        if _valid_arm_chain(
            shoulders[index],
            elbows[index],
            wrists[index],
            humerus_px,
            forearm_px,
            landmark_quality[index],
        )
    ]
    geometry_coverage = len(valid_indices) / len(indices)
    if geometry_coverage < 0.25:
        return "The elbow geometry is too inconsistent across the analyzed set for a full anatomical score."

    max_gap = 0
    current_gap = 0
    valid_set = set(valid_indices)
    for index in indices:
        if index in valid_set:
            max_gap = max(max_gap, current_gap)
            current_gap = 0
        else:
            current_gap += 1
    max_gap = max(max_gap, current_gap)
    if max_gap > max(5, int(round(fps * 0.45))):
        return "A long joint-tracking gap prevents a reliable anatomical score."

    torso_lengths = [
        float(np.linalg.norm(hips[index] - shoulders[index]))
        for index in valid_indices
        if _valid_point(shoulders[index]) and _valid_point(hips[index])
    ]
    if not torso_lengths:
        return "No stable shoulder-to-hip scale was available for an anatomical score."
    torso_lengths = np.asarray(torso_lengths, dtype=float)
    torso_median = max(1.0, float(np.median(torso_lengths)))
    torso_spread = float(np.percentile(torso_lengths, 90) - np.percentile(torso_lengths, 10)) / torso_median
    if torso_spread > 0.45:
        return "The body scale changes too much during the set for reliable joint angles."
    return None


# ---------------------------------------------------------------------------
# LAYER 3 — INDEPENDENT EQUIPMENT TRACKER (HOUGH + LUCAS-KANADE)
# ---------------------------------------------------------------------------

def _reference_plate_candidates(gray: np.ndarray, limit: int = 6) -> list[tuple[np.ndarray, float]]:
    """Return distinct plausible plate circles from frame zero."""
    height, width = gray.shape[:2]
    search_height = max(1, int(round(height * 0.64)))
    cropped = gray[:search_height, :]
    blurred = cv2.GaussianBlur(cropped, (11, 11), 2.0)
    min_radius = max(14, int(round(width * 0.040)))
    max_radius = max(min_radius + 4, int(round(width * 0.26)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(34, int(round(width * 0.075))),
        param1=100,
        param2=36,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    candidates: list[tuple[float, np.ndarray, float]] = []
    for rank, circle in enumerate(circles[0]):
        center = np.array([float(circle[0]), float(circle[1])], dtype=float)
        radius = float(circle[2])
        if not (width * 0.08 <= center[0] <= width * 0.94):
            continue
        if not (height * 0.07 <= center[1] <= height * 0.58):
            continue
        # Hough order carries accumulator strength. The remaining terms prefer
        # a large upper-frame plate without hard-coding one camera position.
        score = (
            rank * 0.08
            + abs(float(center[1]) - height * 0.30) / max(1.0, height) * 0.40
            + abs(float(center[0]) - width * 0.50) / max(1.0, width) * 0.15
            - radius / max(1.0, width) * 0.25
        )
        candidates.append((score, center, radius))
    selected: list[tuple[np.ndarray, float]] = []
    for _, center, radius in sorted(candidates, key=lambda item: item[0]):
        same_object = any(
            float(np.linalg.norm(center - existing_center)) <= max(28.0, min(radius, existing_radius) * 0.55)
            for existing_center, existing_radius in selected
        )
        if same_object:
            continue
        selected.append((center, radius))
        if len(selected) >= limit:
            break
    return selected


def _detect_reference_plate(gray: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    """Acquire the strongest plausible loaded plate in frame zero."""
    candidates = _reference_plate_candidates(gray, limit=1)
    return candidates[0] if candidates else None


def _select_reference_plate(
    gray_frames: list[np.ndarray],
    fps: float,
    *,
    dumbbell_mode: bool = False,
) -> Optional[tuple[np.ndarray, float]]:
    """Select the frame-zero circle with dominant press-like early motion."""
    if not gray_frames:
        return None
    candidates = _reference_plate_candidates(gray_frames[0])
    if not candidates:
        return None
    height, width = gray_frames[0].shape[:2]
    probe_count = min(len(gray_frames), max(18, int(round(fps * 1.8))))
    probe_frames = gray_frames[:probe_count]
    probe_roi = (0, 0, width, max(1, int(round(height * 0.72))))
    ranked: list[tuple[float, np.ndarray, float]] = []
    for rank, candidate in enumerate(candidates):
        path, confidence, _ = _track_equipment(
            probe_frames,
            probe_roi,
            candidate[1] * 2.2,
            [None] * probe_count,
            None,
            candidate,
        )
        direct = np.asarray(
            [
                point
                for point, quality in zip(path, confidence)
                if quality >= 0.70 and _valid_point(point)
            ],
            dtype=float,
        )
        if len(direct) < max(8, int(round(probe_count * 0.55))):
            continue
        vertical_span = float(np.percentile(direct[:, 1], 90) - np.percentile(direct[:, 1], 10))
        horizontal_span = float(np.percentile(direct[:, 0], 90) - np.percentile(direct[:, 0], 10))
        minimum_motion = max(10.0, candidate[1] * 0.12)
        if vertical_span < minimum_motion:
            continue
        directionality = min(1.0, vertical_span / max(8.0, horizontal_span))
        coverage = len(direct) / probe_count
        score = vertical_span * directionality * coverage - rank * 1.5
        ranked.append((score, candidate[0], candidate[1]))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    if len(ranked) > 1 and ranked[1][0] >= ranked[0][0] * 0.90:
        first, second = ranked[0], ranked[1]
        radius_ratio = min(first[2], second[2]) / max(first[2], second[2])
        horizontal_gap = abs(float(first[1][0]) - float(second[1][0]))
        vertical_gap = abs(float(first[1][1]) - float(second[1][1]))
        pair_center_x = (float(first[1][0]) + float(second[1][0])) / 2.0
        pair_radius = max(first[2], second[2])
        symmetric_dumbbell_pair = (
            radius_ratio >= (0.58 if dumbbell_mode else 0.72)
            and horizontal_gap >= pair_radius * (0.75 if dumbbell_mode else 1.15)
            and vertical_gap <= pair_radius * (2.0 if dumbbell_mode else 1.20)
            and abs(pair_center_x - width / 2.0) <= width * (0.34 if dumbbell_mode else 0.22)
        )
        relaxed_dumbbell_pair = (
            dumbbell_mode
            and radius_ratio >= 0.42
            and horizontal_gap >= pair_radius * 0.30
            and vertical_gap <= pair_radius * 3.50
            and max(float(first[1][1]), float(second[1][1])) <= height * 0.90
        )
        if not symmetric_dumbbell_pair and not relaxed_dumbbell_pair and not dumbbell_mode:
            raise RuntimeError(
                "Two different frame-1 circles show similarly strong press-like movement, "
                "so the exercised plate cannot be selected unambiguously."
            )
        # For dumbbells, either member of a validated symmetric pair is a
        # valid first-frame anchor. When the camera perspective prevents the
        # pair geometry test from passing, the highest-ranked moving candidate
        # is still a valid single-arm anchor; later radius/motion validation
        # remains mandatory. Non-dumbbell ties still fail closed.
    return ranked[0][1].copy(), float(ranked[0][2])


def _auto_trim_reference(
    gray_frames: list[np.ndarray],
    fps: float,
) -> tuple[int, Optional[tuple[np.ndarray, float]]]:
    """Find the first early frame whose plate can be validated as a reference."""
    try:
        direct = _select_reference_plate(gray_frames, fps)
    except RuntimeError:
        direct = None
    if direct is not None:
        return 0, direct
    if not gray_frames:
        return 0, None
    search_limit = min(len(gray_frames) - 1, max(1, int(round(fps * 3.0))))
    for start_frame in range(1, search_limit + 1):
        if not _reference_plate_candidates(gray_frames[start_frame], limit=1):
            continue
        try:
            reference = _select_reference_plate(gray_frames[start_frame:], fps)
        except RuntimeError:
            reference = None
        if reference is not None:
            return start_frame, reference
    return 0, None


def _detect_plate_near_reference(
    gray: np.ndarray,
    expected: np.ndarray,
    reference_radius: float,
    previous_radius: float,
    *,
    relaxed: bool = False,
) -> Optional[tuple[np.ndarray, float, float]]:
    """Find the same plate again using strict frame-to-frame continuity."""
    height, width = gray.shape[:2]
    cropped = gray[:max(1, int(round(height * 0.64))), :]
    blurred = cv2.GaussianBlur(cropped, (11, 11), 2.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(34, int(round(width * 0.075))),
        param1=100,
        param2=30 if relaxed else 36,
        minRadius=max(14, int(round(width * 0.040))),
        maxRadius=max(18, int(round(width * 0.26))),
    )
    if circles is None:
        return None

    maximum_shift = max(45.0 if relaxed else 34.0, reference_radius * (0.52 if relaxed else 0.40))
    minimum_ratio, maximum_ratio = ((0.65, 1.55) if relaxed else (0.72, 1.48))
    candidates: list[tuple[float, np.ndarray, float, int]] = []
    for rank, circle in enumerate(circles[0]):
        center = np.array([float(circle[0]), float(circle[1])], dtype=float)
        radius = float(circle[2])
        shift = float(np.linalg.norm(center - expected))
        reference_ratio = radius / max(reference_radius, 1.0)
        previous_ratio = radius / max(previous_radius, 1.0)
        previous_bounds = (0.55, 1.65) if relaxed else (0.65, 1.50)
        if (
            shift > maximum_shift
            or not minimum_ratio <= reference_ratio <= maximum_ratio
            or not previous_bounds[0] <= previous_ratio <= previous_bounds[1]
        ):
            continue
        score = (
            shift / max(reference_radius, 1.0)
            + abs(math.log(max(radius, 1.0) / max(previous_radius, 1.0))) * 0.35
            + rank * (0.035 if relaxed else 0.025)
        )
        candidates.append((score, center, radius, rank))
    if not candidates:
        return None
    _, center, radius, rank = min(candidates, key=lambda item: item[0])
    confidence = 1.0 if rank < 4 else 0.80
    return center, radius, confidence

def _chest_roi(anchor_shoulder: np.ndarray, anchor_hip: np.ndarray, arm_reach_px: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Restrict all plate search to the selected lifter's chest-side workspace."""
    chest = anchor_shoulder * 0.65 + anchor_hip * 0.35
    half_width = int(max(90.0, arm_reach_px * 1.65))
    roi_height = int(max(120.0, arm_reach_px * 2.25))
    x = int(chest[0] - half_width)
    y = int(chest[1] - arm_reach_px * 1.45)
    return _clip_rect(x, y, half_width * 2, roi_height, width, height)


def _detect_plate(
    gray: np.ndarray,
    roi: tuple[int, int, int, int],
    arm_reach_px: float,
    expected: Optional[np.ndarray] = None,
    wrist_reference: Optional[np.ndarray] = None,
    shoulder_anchor: Optional[np.ndarray] = None,
) -> Optional[tuple[np.ndarray, float]]:
    x, y, width, height = roi
    if width < 30 or height < 30:
        return None
    cropped = gray[y:y + height, x:x + width]
    blurred = cv2.GaussianBlur(cropped, (9, 9), 2.0)
    min_radius = max(8, int(arm_reach_px * 0.10))
    max_radius = max(min_radius + 3, int(arm_reach_px * 0.70))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(arm_reach_px * 0.32)),
        param1=80,
        param2=22,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    options = [
        (np.array([float(circle[0] + x), float(circle[1] + y)]), float(circle[2]))
        for circle in circles[0]
    ]
    def association_score(item: tuple[np.ndarray, float]) -> float:
        center, radius = item
        score = -radius * 0.04
        if expected is not None:
            score += float(np.linalg.norm(center - expected)) / max(1.0, arm_reach_px)
        if wrist_reference is not None:
            wrist_distance = float(np.linalg.norm(center - wrist_reference))
            if wrist_distance > arm_reach_px * 1.65:
                return float("inf")
            score += wrist_distance / max(1.0, arm_reach_px) * 0.35
        if wrist_reference is not None and shoulder_anchor is not None:
            arm_vector = wrist_reference - shoulder_anchor
            plate_vector = center - shoulder_anchor
            arm_norm, plate_norm = float(np.linalg.norm(arm_vector)), float(np.linalg.norm(plate_vector))
            if plate_norm > arm_reach_px * 2.25:
                return float("inf")
            if arm_norm > 8.0 and plate_norm > 8.0:
                alignment = float(np.dot(arm_vector, plate_vector) / (arm_norm * plate_norm))
                if alignment < -0.20:
                    return float("inf")
                score += (1.0 - alignment) * 0.45
        return score

    viable = [(association_score(option), option) for option in options]
    viable = [item for item in viable if math.isfinite(item[0])]
    return min(viable, key=lambda item: item[0])[1] if viable else None


def _wrist_reference(wrists: list[Optional[np.ndarray]], index: int, max_gap: int) -> Optional[np.ndarray]:
    if _valid_point(wrists[index]):
        return wrists[index]
    for offset in range(1, max_gap + 1):
        for candidate_index in (index - offset, index + offset):
            if 0 <= candidate_index < len(wrists) and _valid_point(wrists[candidate_index]):
                return wrists[candidate_index]
    return None


def _wrist_observation(
    candidates: list[PoseCandidate],
    wrist_index: int,
    expected: Optional[np.ndarray] = None,
    maximum_distance: float = float("inf"),
) -> Optional[tuple[np.ndarray, float]]:
    options: list[tuple[float, np.ndarray, float]] = []
    for candidate in candidates:
        point = candidate.points.get(wrist_index)
        visibility = candidate.visibility.get(wrist_index, 0.0)
        if visibility < 0.25 or not _valid_point(point):
            continue
        distance = float(np.linalg.norm(point - expected)) if expected is not None else 0.0
        if distance <= maximum_distance:
            options.append((distance - visibility * 12.0, point, visibility))
    if not options:
        return None
    _, seed, _ = min(options, key=lambda item: item[0])
    cluster = [
        (point, visibility)
        for _, point, visibility in options
        if float(np.linalg.norm(point - seed)) <= 55.0
    ]
    center = np.median(np.asarray([item[0] for item in cluster], dtype=float), axis=0)
    return center, max(item[1] for item in cluster)


def _track_dumbbell_handle(
    gray_frames: list[np.ndarray],
    candidates_by_frame: list[list[PoseCandidate]],
    requested_side: Optional[str],
) -> tuple[list[Optional[np.ndarray]], list[float], list[float], str, Optional[PersonTrack]]:
    """Track one dumbbell through its hand/handle instead of circular Hough cues."""
    count = len(gray_frames)
    empty = ([None] * count, [0.0] * count, [0.0] * count)
    if not gray_frames or len(candidates_by_frame) != count:
        return *empty, requested_side or "unknown", None

    height, width = gray_frames[0].shape[:2]
    wrist_indices = {"left": LEFT_WRIST, "right": RIGHT_WRIST}
    if not candidates_by_frame[0]:
        return *empty, requested_side or "unknown", None
    initial_lifter = max(
        candidates_by_frame[0],
        key=lambda candidate: (
            candidate.lower_y / max(1.0, height)
            + candidate.body_scale / max(1.0, width)
        ),
    )
    if requested_side in wrist_indices:
        side = requested_side
    else:
        side = max(
            wrist_indices,
            key=lambda candidate_side: initial_lifter.visibility.get(wrist_indices[candidate_side], 0.0),
        )

    wrist_index = wrist_indices[side]
    initial = _wrist_observation([initial_lifter], wrist_index)
    if initial is None:
        return *empty, side, None

    handle_radius = max(18.0, width * 0.045)
    path: list[Optional[np.ndarray]] = [None] * count
    confidence: list[float] = [0.0] * count
    radii: list[float] = [0.0] * count
    current = initial[0].copy()
    lifter_track = PersonTrack(track_id=0)
    lifter_track.append(initial_lifter)
    lifter_center = initial_lifter.center.copy()
    lifter_scale = float(initial_lifter.body_scale)
    path[0], confidence[0], radii[0] = current.copy(), max(0.72, initial[1]), handle_radius
    lk_params = dict(
        winSize=(41, 41),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    observation_gate = max(120.0, width * 0.25)

    for index in range(1, count):
        next_point, status, error = cv2.calcOpticalFlowPyrLK(
            gray_frames[index - 1],
            gray_frames[index],
            np.asarray([[current]], dtype=np.float32),
            None,
            **lk_params,
        )
        flow_ok = (
            next_point is not None
            and status is not None
            and int(status.ravel()[0]) == 1
            and error is not None
            and float(error.ravel()[0]) < 35.0
        )
        predicted = next_point.reshape(2).astype(float) if flow_ok else None
        expected = predicted if predicted is not None else current
        associated: list[tuple[float, PoseCandidate]] = []
        for candidate in candidates_by_frame[index]:
            center_distance = float(np.linalg.norm(candidate.center - lifter_center))
            scale_ratio = max(candidate.body_scale, lifter_scale) / max(
                1.0,
                min(candidate.body_scale, lifter_scale),
            )
            point = candidate.points.get(wrist_index)
            if (
                center_distance <= max(90.0, lifter_scale * 1.9)
                and scale_ratio < 2.25
                and _valid_point(point)
            ):
                wrist_distance = float(np.linalg.norm(point - expected))
                if wrist_distance <= observation_gate:
                    associated.append(
                        (
                            center_distance / max(1.0, lifter_scale)
                            + wrist_distance / max(1.0, observation_gate) * 0.45,
                            candidate,
                        )
                    )
        selected_candidate = min(associated, key=lambda item: item[0])[1] if associated else None
        observation = (
            _wrist_observation([selected_candidate], wrist_index, expected, observation_gate)
            if selected_candidate is not None
            else None
        )
        if observation is not None:
            observed, visibility = observation
            if predicted is not None and float(np.linalg.norm(observed - predicted)) <= 90.0:
                current = observed * 0.65 + predicted * 0.35
            else:
                current = observed.copy()
            quality = max(0.72, visibility)
            lifter_center = selected_candidate.center.copy()
            lifter_scale = float(selected_candidate.body_scale)
            if selected_candidate.frame_index > lifter_track.last_frame:
                lifter_track.append(selected_candidate)
        elif predicted is not None:
            current = predicted
            quality = 0.74
        else:
            continue
        if not (0.0 <= current[0] < width and 0.0 <= current[1] < height):
            continue
        path[index], confidence[index], radii[index] = current.copy(), quality, handle_radius
    return path, confidence, radii, side, lifter_track


def _track_barbell_grip(
    gray_frames: list[np.ndarray],
    lifter_track: PersonTrack,
    candidates_by_frame: list[list[PoseCandidate]],
) -> tuple[list[Optional[np.ndarray]], list[float], list[float], Optional[PersonTrack]]:
    """Fallback for visible bars whose plates are not reliable circular detections."""
    count = len(gray_frames)
    path: list[Optional[np.ndarray]] = [None] * count
    confidence: list[float] = [0.0] * count
    radii: list[float] = [0.0] * count
    if not gray_frames or not lifter_track.samples or len(candidates_by_frame) != count:
        return path, confidence, radii, None
    height, width = gray_frames[0].shape[:2]

    def grip_observation(candidate: PoseCandidate) -> Optional[tuple[np.ndarray, float, float]]:
        left = candidate.points.get(LEFT_WRIST)
        right = candidate.points.get(RIGHT_WRIST)
        left_quality = candidate.visibility.get(LEFT_WRIST, 0.0)
        right_quality = candidate.visibility.get(RIGHT_WRIST, 0.0)
        if (
            left_quality < 0.35
            or right_quality < 0.35
            or not _valid_point(left)
            or not _valid_point(right)
        ):
            return None
        hand_span = float(np.linalg.norm(right - left))
        if hand_span < max(20.0, width * 0.06):
            return None
        return (left + right) / 2.0, min(left_quality, right_quality), hand_span

    samples_by_frame = {sample.frame_index: sample.candidate for sample in lifter_track.samples}
    first_candidate = samples_by_frame.get(0)
    if first_candidate is None:
        return path, confidence, radii, None
    initial = grip_observation(first_candidate)
    if initial is None:
        return path, confidence, radii, None
    confirmed_track = PersonTrack(track_id=lifter_track.track_id)
    confirmed_track.append(first_candidate)
    current, initial_quality, initial_span = initial
    path[0] = current.copy()
    confidence[0] = initial_quality
    radii[0] = max(16.0, initial_span * 0.14)
    lifter_center = first_candidate.center.copy()
    lifter_scale = float(first_candidate.body_scale)
    observation_gate = max(110.0, width * 0.24)

    for index in range(1, count):
        preferred = samples_by_frame.get(index)
        candidates: list[tuple[float, PoseCandidate, tuple[np.ndarray, float, float]]] = []
        pool = [preferred] if preferred is not None else candidates_by_frame[index]
        for candidate in pool:
            if candidate is None:
                continue
            observation = grip_observation(candidate)
            if observation is None:
                continue
            center_distance = float(np.linalg.norm(candidate.center - lifter_center))
            scale_ratio = max(candidate.body_scale, lifter_scale) / max(
                1.0, min(candidate.body_scale, lifter_scale)
            )
            grip_distance = float(np.linalg.norm(observation[0] - current))
            if (
                center_distance <= max(90.0, lifter_scale * 1.9)
                and scale_ratio < 2.25
                and grip_distance <= observation_gate
            ):
                candidates.append(
                    (
                        center_distance / max(1.0, lifter_scale)
                        + grip_distance / max(1.0, observation_gate) * 0.45,
                        candidate,
                        observation,
                    )
                )
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        if (
            preferred is None
            and len(candidates) > 1
            and candidates[1][0] - candidates[0][0] < 0.15
        ):
            continue
        _, selected, observation = candidates[0]
        current, quality, hand_span = observation
        lifter_center = selected.center.copy()
        lifter_scale = float(selected.body_scale)
        confirmed_track.append(selected)
        path[index] = current.copy()
        confidence[index] = quality
        radii[index] = max(16.0, hand_span * 0.14)
    return path, confidence, radii, confirmed_track


def _grip_track_is_reliable(
    path: list[Optional[np.ndarray]],
    confidence: list[float],
    fps: float,
) -> bool:
    """Require continuous, directly observed two-hand grip evidence."""
    direct = [
        index
        for index, (point, quality) in enumerate(zip(path, confidence))
        if _valid_point(point) and quality >= 0.70
    ]
    if len(direct) < max(8, int(round(fps * 0.60))) or len(direct) / max(1, len(path)) < 0.70:
        return False
    # Leading/trailing loss happens before setup or after racking the bar and
    # does not invalidate an otherwise direct observation sequence. Internal
    # gaps must remain short enough to bridge only between real observations.
    internal_gaps = [right - left - 1 for left, right in zip(direct, direct[1:])]
    return max(internal_gaps, default=0) <= max(5, int(round(fps * 0.20)))


def _track_equipment(
    gray_frames: list[np.ndarray],
    roi: tuple[int, int, int, int],
    arm_reach_px: float,
    wrists: list[Optional[np.ndarray]],
    shoulder_anchor: Optional[np.ndarray],
    initial_detection: Optional[tuple[np.ndarray, float]] = None,
) -> tuple[list[Optional[np.ndarray]], list[float], list[float]]:
    """Track one plate with per-frame Hough confirmation and LK fallback."""
    count = len(gray_frames)
    path: list[Optional[np.ndarray]] = [None] * count
    confidence: list[float] = [0.0] * count
    radii: list[float] = [0.0] * count
    if not gray_frames:
        return path, confidence, radii

    detection = initial_detection or _detect_reference_plate(gray_frames[0])
    if detection is None:
        return path, confidence, radii
    current = np.asarray(detection[0], dtype=float).copy()
    reference_radius = float(detection[1])
    radius = reference_radius
    path[0], confidence[0], radii[0] = current.copy(), 1.0, radius

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.02),
    )
    for index in range(1, count):
        previous = gray_frames[index - 1]
        current_frame = gray_frames[index]
        circle = _detect_plate_near_reference(
            current_frame,
            current,
            reference_radius,
            radius,
        )
        if circle is None:
            circle = _detect_plate_near_reference(
                current_frame,
                current,
                reference_radius,
                radius,
                relaxed=True,
            )
        if circle is not None:
            current, radius, circle_confidence = circle
            path[index] = current.copy()
            confidence[index] = circle_confidence
            radii[index] = radius
            continue

        # Hough is authoritative. LK only bridges an isolated frame and stays
        # below the scoring gate until a circle is confirmed again.
        next_point, status, error = cv2.calcOpticalFlowPyrLK(
            previous,
            current_frame,
            np.asarray([[current]], dtype=np.float32),
            None,
            **lk_params,
        )
        tracking_ok = (
            next_point is not None
            and status is not None
            and int(status.ravel()[0]) == 1
            and error is not None
            and float(error.ravel()[0]) < 25.0
        )
        candidate = next_point.reshape(2).astype(float) if tracking_ok else None
        x, y, width, height = roi
        in_roi = (
            candidate is not None
            and x <= candidate[0] <= x + width
            and y <= candidate[1] <= y + height
            and float(np.linalg.norm(candidate - current)) <= max(24.0, reference_radius * 0.30)
        )
        if not in_roi:
            continue
        current = candidate
        path[index], confidence[index], radii[index] = current.copy(), 0.75, radius
    return path, confidence, radii


def _bar_shaft_support_ratio(
    gray_frames: list[np.ndarray],
    equipment_path: list[Optional[np.ndarray]],
    equipment_confidence: list[float],
    equipment_radii: list[float],
    track: PersonTrack,
) -> float:
    """Measure repeated long image edges from one plate through both hands."""
    candidates = {
        sample.frame_index: sample.candidate
        for sample in track.samples
        if sample.candidate is not None
    }
    usable = [
        index
        for index, (point, quality) in enumerate(
            zip(equipment_path, equipment_confidence)
        )
        if index in candidates
        and _valid_point(point)
        and quality >= 0.55
        and index < len(gray_frames)
        and index < len(equipment_radii)
    ]
    if not usable:
        return 0.0
    step = max(1, len(usable) // 16)
    sampled = usable[::step][:16]
    supported = 0
    evaluated = 0
    for index in sampled:
        candidate = candidates[index]
        left = candidate.points.get(LEFT_WRIST)
        right = candidate.points.get(RIGHT_WRIST)
        if (
            candidate.visibility.get(LEFT_WRIST, 0.0) < 0.35
            or candidate.visibility.get(RIGHT_WRIST, 0.0) < 0.35
            or not _valid_point(left)
            or not _valid_point(right)
        ):
            continue
        plate = np.asarray(equipment_path[index], dtype=float)
        wrists = [
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
        ]
        far_wrist = max(wrists, key=lambda wrist: np.linalg.norm(wrist - plate))
        direction = far_wrist - plate
        distance = float(np.linalg.norm(direction))
        if distance < 36.0:
            continue
        unit = direction / distance
        margin = max(14, int(round(candidate.body_scale * 0.10)))
        x0 = max(0, int(np.floor(min(plate[0], far_wrist[0]) - margin)))
        y0 = max(0, int(np.floor(min(plate[1], far_wrist[1]) - margin)))
        x1 = min(
            gray_frames[index].shape[1],
            int(np.ceil(max(plate[0], far_wrist[0]) + margin)),
        )
        y1 = min(
            gray_frames[index].shape[0],
            int(np.ceil(max(plate[1], far_wrist[1]) + margin)),
        )
        if x1 - x0 < 24 or y1 - y0 < 16:
            continue
        edges = cv2.Canny(gray_frames[index][y0:y1, x0:x1], 45, 130)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=16,
            minLineLength=max(18, int(round(distance * 0.20))),
            maxLineGap=12,
        )
        evaluated += 1
        if lines is None:
            continue
        for raw_line in np.asarray(lines).reshape(-1, 4):
            start = np.array(
                [float(raw_line[0] + x0), float(raw_line[1] + y0)]
            )
            end = np.array(
                [float(raw_line[2] + x0), float(raw_line[3] + y0)]
            )
            line = end - start
            length = float(np.linalg.norm(line))
            if length <= 0.0 or abs(float(np.dot(line / length, unit))) < 0.94:
                continue
            midpoint = (start + end) / 2.0
            offset = midpoint - plate
            perpendicular = abs(
                float(unit[0] * offset[1] - unit[1] * offset[0])
            )
            if perpendicular > margin:
                continue
            projections = sorted(
                [
                    float(np.dot(start - plate, unit)),
                    float(np.dot(end - plate, unit)),
                ]
            )
            maximum_start = max(
                distance * 0.25,
                float(equipment_radii[index]) * 1.25,
            )
            if projections[0] <= maximum_start and projections[1] >= distance * 0.65:
                supported += 1
                break
    return supported / max(1, evaluated)


def _validate_equipment_track(
    path: list[Optional[np.ndarray]],
    confidence: list[float],
    radii: list[float],
    reference: FrameReference,
    fps: float,
) -> None:
    """Reject a persistent but static/background circle before scoring it."""
    direct_indices = [
        index
        for index, (point, quality) in enumerate(zip(path, confidence))
        if quality >= 0.70 and _valid_point(point)
    ]
    if len(direct_indices) < max(8, int(round(fps * 0.60))):
        raise RuntimeError("The first-frame plate could not be confirmed through enough of the recording.")
    direct_points = np.asarray([path[index] for index in direct_indices], dtype=float)
    vertical_span = float(np.percentile(direct_points[:, 1], 90) - np.percentile(direct_points[:, 1], 10))
    if vertical_span < max(14.0, reference.plate_radius * 0.18):
        raise RuntimeError(
            "The circle selected in frame 1 remained effectively static, so it was rejected as possible background equipment."
        )
    direct_radii = np.asarray([radii[index] for index in direct_indices if radii[index] > 0.0], dtype=float)
    if len(direct_radii) < max(5, int(round(fps * 0.30))):
        raise RuntimeError("The tracked object was not repeatedly confirmed as a circular plate.")
    median_radius = float(np.median(direct_radii))
    if not reference.plate_radius * 0.72 <= median_radius <= reference.plate_radius * 1.48:
        raise RuntimeError("The tracked circle changed scale too much to be the frame-1 plate.")


def _interpolate_short_gaps(points: list[Optional[np.ndarray]], max_gap: int) -> list[Optional[np.ndarray]]:
    result = list(points)
    index = 0
    while index < len(result):
        if _valid_point(result[index]):
            index += 1
            continue
        start = index
        while index < len(result) and not _valid_point(result[index]):
            index += 1
        end = index
        if start == 0 or end == len(result) or end - start > max_gap:
            continue
        before, after = result[start - 1], result[end]
        for gap_index in range(start, end):
            fraction = (gap_index - start + 1) / (end - start + 1)
            result[gap_index] = before * (1.0 - fraction) + after * fraction
    return result


def _smooth_points(points: list[Optional[np.ndarray]], window: int = 5) -> list[Optional[np.ndarray]]:
    """Smooth valid runs while preserving unsupported long gaps as missing."""
    result = list(points)
    start = 0
    while start < len(points):
        while start < len(points) and not _valid_point(points[start]):
            start += 1
        end = start
        while end < len(points) and _valid_point(points[end]):
            end += 1
        if end - start >= 3:
            run = np.asarray(points[start:end], dtype=float)
            result[start:end] = [np.array([_smooth_1d(run[:, dim], window)[row] for dim in range(2)]) for row in range(len(run))]
        start = end
    return result


# ---------------------------------------------------------------------------
# LAYER 4 — DYNAMIC PERSPECTIVE PROJECTION
# ---------------------------------------------------------------------------

def _project_template_x(
    template_point: np.ndarray,
    reference_template: np.ndarray,
    calibration: PerspectiveCalibration,
    reference_bar_x: float,
) -> float:
    """Rotate the 3D chest template around camera yaw and map it into image X."""
    yaw = calibration.yaw_radians
    projected = template_point[0] * math.cos(yaw) + template_point[2] * math.sin(yaw)
    projected_reference = reference_template[0] * math.cos(yaw) + reference_template[2] * math.sin(yaw)
    return reference_bar_x + calibration.facing_direction * (projected - projected_reference) * calibration.scale_px_per_m


def _phase_at(index: int, y_values: np.ndarray) -> str:
    previous = max(0, index - 2)
    while previous > 0 and not math.isfinite(float(y_values[previous])):
        previous -= 1
    if not math.isfinite(float(y_values[index])) or not math.isfinite(float(y_values[previous])):
        return "concentric"
    # Screen Y increases downwards: descending bar is eccentric on a press.
    return "eccentric" if y_values[index] > y_values[previous] else "concentric"


# ---------------------------------------------------------------------------
# LAYER 5 — REPETITIONS, SCORING AND RENDERING
# ---------------------------------------------------------------------------

def _detect_reps(
    bar_path: list[Optional[np.ndarray]],
    bar_confidence: list[float],
    fps: float,
    confidence_gate: float = 0.70,
) -> tuple[list[Rep], np.ndarray]:
    """Find only fully measured top -> bottom -> top repetitions.

    Long tracking gaps remain missing and split the video into independent
    segments. They are never converted to a median-valued bridge.
    """
    raw_y = np.asarray(
        [
            point[1] if _valid_point(point) and confidence >= confidence_gate else np.nan
            for point, confidence in zip(bar_path, bar_confidence)
        ],
        dtype=float,
    )
    if np.count_nonzero(np.isfinite(raw_y)) < max(8, int(fps * 0.8)):
        return [], raw_y
    filled = raw_y.copy()
    measured = np.isfinite(raw_y)
    max_short_gap = max(2, int(fps * 0.20))
    index = 0
    while index < len(filled):
        if math.isfinite(float(filled[index])):
            index += 1
            continue
        gap_start = index
        while index < len(filled) and not math.isfinite(float(filled[index])):
            index += 1
        gap_end = index
        if gap_start == 0 or gap_end == len(filled) or gap_end - gap_start > max_short_gap:
            continue
        before, after = filled[gap_start - 1], filled[gap_end]
        for gap_index in range(gap_start, gap_end):
            fraction = (gap_index - gap_start + 1) / (gap_end - gap_start + 1)
            filled[gap_index] = before * (1.0 - fraction) + after * fraction

    reps: list[Rep] = []
    segment_start = 0
    while segment_start < len(filled):
        while segment_start < len(filled) and not math.isfinite(float(filled[segment_start])):
            segment_start += 1
        segment_end = segment_start
        while segment_end < len(filled) and math.isfinite(float(filled[segment_end])):
            segment_end += 1
        if segment_end - segment_start < max(8, int(fps * 0.8)):
            segment_start = segment_end
            continue

        segment = _smooth_1d(filled[segment_start:segment_end], max(5, int(fps // 5) | 1))
        span = float(np.percentile(segment, 90) - np.percentile(segment, 10))
        if span < 8.0:
            segment_start = segment_end
            continue
        distance = max(3, int(fps * 0.45))
        bottoms = [segment_start + item for item in _turning_points(segment, distance, span * 0.18, maxima=True)]
        tops = [segment_start + item for item in _turning_points(segment, distance, span * 0.18, maxima=False)]
        top_gate = float(np.percentile(segment, 35))
        # A controlled lockout often forms a broad plateau rather than one
        # sharp minimum. Recover one direct top observation between consecutive
        # bottoms so two real repetitions are not merged into one long cycle.
        for earlier_bottom, later_bottom in zip(bottoms, bottoms[1:]):
            if later_bottom - earlier_bottom < max(4, int(fps * 0.30)):
                continue
            local_start = earlier_bottom - segment_start + 1
            local_end = later_bottom - segment_start
            if local_end <= local_start:
                continue
            local_top = local_start + int(np.argmin(segment[local_start:local_end]))
            candidate_top = segment_start + local_top
            if segment[local_top] <= top_gate and measured[candidate_top]:
                tops.append(candidate_top)
        tops = sorted(set(tops))
        if bottoms:
            first_bottom = bottoms[0]
            local_end = first_bottom - segment_start
            if local_end >= max(4, int(fps * 0.30)):
                initial_top_local = int(np.argmin(segment[:local_end]))
                initial_top = segment_start + initial_top_local
                if (
                    segment[initial_top_local] <= top_gate
                    and measured[initial_top]
                    and first_bottom - initial_top >= max(4, int(fps * 0.30))
                ):
                    tops.append(initial_top)
                    tops = sorted(set(tops))
        if measured[segment_start] and segment[0] <= top_gate:
            tops.insert(0, segment_start)
        if measured[segment_end - 1] and segment[-1] <= top_gate:
            tops.append(segment_end - 1)

        used_top_end = segment_start - 1
        for bottom in bottoms:
            earlier = [top for top in tops if top < bottom and top >= used_top_end]
            later = [top for top in tops if top > bottom]
            if not earlier or not later:
                continue
            start, end = earlier[-1], later[0]
            # All three repetition extrema must be direct equipment readings.
            if not (measured[start] and measured[bottom] and measured[end]):
                continue
            if not (int(fps * 0.30) <= end - start <= int(fps * 12.0)):
                continue
            local = segment[start - segment_start:end - segment_start + 1]
            local_rom = float(np.max(local) - np.min(local))
            if local_rom < span * 0.55:
                continue
            reps.append(Rep(start, bottom, end, local_rom))
            used_top_end = end
        segment_start = segment_end
    return reps, filled


def _filter_consistent_reps(reps: list[Rep], y_values: np.ndarray) -> list[Rep]:
    """Keep the repeated set pattern and reject smaller post-set handling cycles."""
    if len(reps) < 2:
        return reps
    median_rom = float(np.median([rep.rom_px for rep in reps]))
    top_levels = [
        (float(y_values[rep.start_frame]) + float(y_values[rep.end_frame])) / 2.0
        for rep in reps
    ]
    median_top = float(np.median(top_levels))
    return [
        rep
        for rep, top_level in zip(reps, top_levels)
        if (
            median_rom * 0.70 <= rep.rom_px <= median_rom * 1.40
            and abs(top_level - median_top) <= max(24.0, median_rom * 0.16)
        )
    ]


def _score_dumbbell_reps(
    reps: list[Rep],
    path: list[Optional[np.ndarray]],
    confidence: list[float],
) -> tuple[dict[int, int], dict[int, float]]:
    """Score dumbbell-path repeatability without pretending anatomy is known."""
    if len(reps) < 2:
        return {}, {}
    samples_per_phase = 17
    profiles: list[np.ndarray] = []
    valid_reps: list[tuple[int, Rep]] = []
    for rep_index, rep in enumerate(reps):
        points = path[rep.start_frame:rep.end_frame + 1]
        if any(not _valid_point(point) for point in points):
            continue
        x_values = np.asarray([point[0] for point in points], dtype=float)
        bottom_local = rep.bottom_frame - rep.start_frame
        if bottom_local < 2 or len(x_values) - bottom_local < 3:
            continue
        eccentric = np.interp(
            np.linspace(0.0, bottom_local, samples_per_phase),
            np.arange(bottom_local + 1),
            x_values[:bottom_local + 1],
        )
        concentric_source = x_values[bottom_local:]
        concentric = np.interp(
            np.linspace(0.0, len(concentric_source) - 1, samples_per_phase),
            np.arange(len(concentric_source)),
            concentric_source,
        )
        top_x = (x_values[0] + x_values[-1]) / 2.0
        y_values = np.asarray([point[1] for point in points], dtype=float)
        eccentric_y = np.interp(
            np.linspace(0.0, bottom_local, samples_per_phase),
            np.arange(bottom_local + 1),
            y_values[:bottom_local + 1],
        )
        concentric_y_source = y_values[bottom_local:]
        concentric_y = np.interp(
            np.linspace(0.0, len(concentric_y_source) - 1, samples_per_phase),
            np.arange(len(concentric_y_source)),
            concentric_y_source,
        )
        top_y = (y_values[0] + y_values[-1]) / 2.0
        normalized_x = np.concatenate([eccentric, concentric[1:]]) - top_x
        normalized_y = np.concatenate([eccentric_y, concentric_y[1:]]) - top_y
        profiles.append(np.column_stack([normalized_x, normalized_y]) / max(1.0, rep.rom_px))
        valid_reps.append((rep_index, rep))
    if len(profiles) < 2:
        return {}, {}

    median_rom = float(np.median([rep.rom_px for _, rep in valid_reps]))
    rep_scores: dict[int, int] = {}
    frame_scores: dict[int, float] = {}
    profile_array = np.asarray(profiles)
    for profile_index, ((rep_index, rep), profile) in enumerate(zip(valid_reps, profiles)):
        other_profiles = np.delete(profile_array, profile_index, axis=0)
        template = np.median(other_profiles, axis=0)
        shape_error = float(np.mean(np.linalg.norm(profile - template, axis=1)))
        rom_error = abs(rep.rom_px - median_rom) / max(1.0, median_rom)
        closure_error = float(np.linalg.norm(profile[0] - profile[-1]))
        rep_score = float(np.clip(
            100.0 - shape_error * 70.0 - rom_error * 25.0 - closure_error * 18.0,
            0.0,
            100.0,
        ))
        rep_scores[rep_index] = int(round(rep_score))
        rep.score = rep_scores[rep_index]
        for frame_index in range(rep.start_frame, rep.end_frame + 1):
            if confidence[frame_index] >= 0.70 and _valid_point(path[frame_index]):
                frame_scores[frame_index] = rep_score
    return rep_scores, frame_scores


def _score_reps(
    reps: list[Rep],
    bar_path: list[Optional[np.ndarray]],
    shoulders: list[Optional[np.ndarray]],
    elbows: list[Optional[np.ndarray]],
    wrists: list[Optional[np.ndarray]],
    elbow_observed_valid: list[bool],
    calibration: PerspectiveCalibration,
    levers: dict,
    trajectory_func: Callable,
    y_values: np.ndarray,
    bar_confidence: list[float],
    reference: FrameReference,
) -> ScoreBreakdown:
    breakdown = ScoreBreakdown()
    if not reps:
        return breakdown
    reference_rep = next(
        (rep for rep in reps if rep.start_frame <= reference.frame_index <= rep.end_frame),
        reps[0],
    )
    if not math.isfinite(float(y_values[reference_rep.bottom_frame])):
        return {}, {}
    reference_progress = float(np.clip(
        (float(y_values[reference_rep.bottom_frame]) - float(reference.bar_center[1]))
        / max(1.0, reference_rep.rom_px),
        0.0,
        1.0,
    ))
    reference_template = trajectory_func(reference_progress, levers, phase="eccentric")
    for rep_index, rep in enumerate(reps):
        bar_scores: list[float] = []
        limb_scores: list[float] = []
        rep_length = rep.end_frame - rep.start_frame + 1
        for frame_index in range(rep.start_frame, rep.end_frame + 1):
            bar = bar_path[frame_index]
            shoulder = shoulders[frame_index]
            elbow = elbows[frame_index]
            wrist = wrists[frame_index]
            if not _valid_point(bar) or bar_confidence[frame_index] < 0.70:
                continue
            progress = np.clip(
                (float(y_values[rep.bottom_frame]) - float(y_values[frame_index])) / max(1.0, rep.rom_px),
                0.0,
                1.0,
            )
            phase = _phase_at(frame_index, y_values)
            template = trajectory_func(float(progress), levers, phase=phase)
            target_x = _project_template_x(
                template,
                reference_template,
                calibration,
                float(reference.bar_center[0]),
            )
            path_penalty = abs(float(bar[0]) - target_x) / max(8.0, calibration.arm_length_px * 0.23)
            bar_score = float(np.clip(100.0 - path_penalty * 34.0, 0.0, 100.0))
            bar_scores.append(bar_score)
            breakdown.bar_frame_scores[frame_index] = bar_score
            if elbow_observed_valid[frame_index] and _valid_point(elbow) and _valid_point(wrist):
                forearm = wrist - elbow
                forearm_length = float(np.linalg.norm(forearm))
                if forearm_length > 8.0:
                    verticality_penalty = abs(float(forearm[0])) / forearm_length
                    limb_score = float(np.clip(100.0 - verticality_penalty * 34.0, 0.0, 100.0))
                    limb_scores.append(limb_score)
                    breakdown.limb_frame_scores[frame_index] = limb_score
        # A few isolated equipment points cannot establish a trajectory score.
        if len(bar_scores) < max(6, int(math.ceil(rep_length * 0.60))):
            continue
        breakdown.bar_rep_scores[rep_index] = int(round(float(np.mean(bar_scores))))
        # Limb scoring is stricter: it must cover most of the same observed bar
        # movement and never fills missing elbow data with a made-up value.
        if len(limb_scores) >= max(6, int(math.ceil(len(bar_scores) * 0.60))):
            breakdown.limb_rep_scores[rep_index] = int(round(float(np.mean(limb_scores))))
            combined = int(round(
                (breakdown.bar_rep_scores[rep_index] + breakdown.limb_rep_scores[rep_index]) / 2.0
            ))
            breakdown.overall_rep_scores[rep_index] = combined
            rep.score = combined
        else:
            rep.score = breakdown.bar_rep_scores[rep_index]
    return breakdown


def _draw_text_panel(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]],
                     origin: tuple[int, int] = (20, 20)) -> None:
    height, width = frame.shape[:2]

    # Dynamiczny mnożnik wielkości – dopasowuje się do szerokości filmu
    scale = max(0.4, min(1.0, width / 900.0))

    x, y = int(origin[0] * scale), int(origin[1] * scale)
    font_scale = 0.68 * scale
    font_thick = max(1, int(2 * scale))
    char_w = 14 * scale

    box_width = int(max(360 * scale, max((len(text) for text, _ in lines), default=0) * char_w + (30 * scale)))
    box_height = int(18 * scale + len(lines) * (34 * scale))

    # Rysowanie lekko przezroczystego tła (60% czerni, 40% widoczności tła)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Renderowanie dynamicznego tekstu
    for line_index, (text, color) in enumerate(lines):
        text_y = y + int(28 * scale) + line_index * int(34 * scale)
        cv2.putText(frame, text, (x + int(14 * scale), text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick,
                    cv2.LINE_AA)


def _score_color(score: float) -> tuple[int, int, int]:
    return (48, 200, 80) if score >= 75 else (0, 180, 255) if score >= 50 else (50, 60, 235)


def _render_output(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float,
    exercise_key: str,
    near_arm: str,
    display_shoulders: list[Optional[np.ndarray]],
    reference: FrameReference,
    bar_path: list[Optional[np.ndarray]],
    bar_confidence: list[float],
    plate_radii: list[float],
    elbows: list[Optional[np.ndarray]],
    hips: list[Optional[np.ndarray]],
    reps: list[Rep],
    bar_frame_scores: dict[int, float],
    limb_frame_scores: dict[int, float],
    bar_path_score: Optional[int],
    limb_motion_score: Optional[int],
    final_score: Optional[int],
    analysis_mode: str,
) -> None:
    height, width = frames[0].shape[:2]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video '{destination}'.")

    completed_by_frame = {rep.end_frame: index + 1 for index, rep in enumerate(reps)}
    completed = 0
    history: list[np.ndarray] = []
    last_rendered = frames[-1].copy()
    for index, source in enumerate(frames):
        frame = source.copy()
        if index in completed_by_frame:
            completed = completed_by_frame[index]
        bar = bar_path[index]
        shoulder = display_shoulders[index]
        elbow = elbows[index]
        hip = hips[index]
        if _valid_point(shoulder) and _valid_point(elbow) and _valid_point(bar):
            cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (0, 165, 255), 3)
            cv2.line(frame, tuple(elbow.astype(int)), tuple(bar.astype(int)), (0, 165, 255), 3)
            cv2.circle(frame, tuple(elbow.astype(int)), 6, (30, 90, 250), -1)
        if _valid_point(shoulder):
            cv2.circle(frame, tuple(shoulder.astype(int)), 7, (255, 255, 255), -1)
        reference_center = tuple(reference.bar_center.astype(int))
        cv2.circle(frame, reference_center, max(10, int(reference.plate_radius)), (70, 220, 90), 2)
        cv2.drawMarker(frame, reference_center, (70, 220, 90), cv2.MARKER_CROSS, 24, 3)
        if _valid_point(bar):
            history.append(bar)
            radius = max(8, int(plate_radii[index])) if index < len(plate_radii) else 9
            cv2.circle(frame, tuple(bar.astype(int)), radius, (30, 80, 245), 2)
            cv2.circle(frame, tuple(bar.astype(int)), 5, (30, 80, 245), -1)
        for history_index in range(1, len(history)):
            cv2.line(frame, tuple(history[history_index - 1].astype(int)), tuple(history[history_index].astype(int)), (255, 185, 45), 2)

        live_bar_score = bar_frame_scores.get(index)
        live_limb_score = limb_frame_scores.get(index)
        quality_note = "TRACKED" if _valid_point(bar) and bar_confidence[index] >= 0.70 else "BAR LOST — EXCLUDED"
        lines = [
            (f"Exercise: {exercise_key}", (245, 245, 245)),
            (f"Frame 1 reference: LOCKED | {quality_note}", (100, 220, 255) if _valid_point(bar) else (40, 80, 255)),
            (f"Mode: {analysis_mode} | Near arm: {near_arm.upper()}", (245, 245, 245)),
            (f"Reps complete: {completed}/{len(reps)}", (245, 245, 245)),
        ]
        if live_bar_score is not None:
            lines.append((f"Bar path: {int(round(live_bar_score))}%", _score_color(live_bar_score)))
        if live_limb_score is not None:
            lines.append((f"Limb motion: {int(round(live_limb_score))}%", _score_color(live_limb_score)))
        elif limb_motion_score is None:
            lines.append(("Limb motion: N/A — anatomy unavailable", (160, 180, 195)))
        _draw_text_panel(frame, lines)
        writer.write(frame)
        last_rendered = frame

    summary = last_rendered.copy()
    summary_lines = [
        ("TECHNIQUE RESULTS", (245, 245, 245)),
        (
            f"BAR PATH SCORE: {bar_path_score}%" if bar_path_score is not None else "BAR PATH SCORE: N/A",
            _score_color(bar_path_score) if bar_path_score is not None else (160, 180, 195),
        ),
        (
            f"LIMB MOTION SCORE: {limb_motion_score}%"
            if limb_motion_score is not None
            else "LIMB MOTION SCORE: N/A",
            _score_color(limb_motion_score) if limb_motion_score is not None else (160, 180, 195),
        ),
        (
            f"FINAL SCORE: {final_score}%" if final_score is not None else "FINAL SCORE: N/A",
            _score_color(final_score) if final_score is not None else (160, 180, 195),
        ),
        (f"COMPLETED REPS: {len(reps)}", (245, 245, 245)),
        (f"ANALYSIS: {analysis_mode}", (180, 210, 230)),
    ]
    overlay = summary.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (10, 10, 10), -1)
    summary = cv2.addWeighted(overlay, 0.72, summary, 0.28, 0)
    line_spacing = min(50, max(36, (height - 60) // max(1, len(summary_lines))))
    block_height = line_spacing * (len(summary_lines) - 1)
    y = max(42, (height - block_height) // 2)
    for text, color in summary_lines:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3 if text == "TECHNIQUE RESULTS" else 0.92, 3)[0]
        x = max(20, (width - text_size[0]) // 2)
        cv2.putText(summary, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3 if text == "TECHNIQUE RESULTS" else 0.92, color, 3, cv2.LINE_AA)
        y += line_spacing
    for _ in range(max(1, int(round(fps * 2.0)))):
        writer.write(summary)
    writer.release()


def _analyze_video_once(
    video_input_path: str | Path,
    video_output_path: str | Path = "vision_analysis.mp4",
    exercise_key: str = "Flat_Barbell_Press",
    *,
    model_path: Optional[str | Path] = None,
    near_arm: Optional[str] = None,
    min_pose_confidence: float = 0.35,
) -> dict:
    """
    Analyze a chest-exercise video and write an annotated result.

    Returns a JSON-serializable summary. A RuntimeError means the source does
    not contain enough trustworthy visual information and no misleading score
    was produced.
    """
    source = Path(video_input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if near_arm not in {None, "left", "right"}:
        raise ValueError("near_arm must be None, 'left', or 'right'.")
    if not 0.05 <= min_pose_confidence <= 0.95:
        raise ValueError("min_pose_confidence must be between 0.05 and 0.95.")

    exercise_spec, levers, _, trajectory_func = _load_exercise_model(exercise_key)
    pose_model = _ensure_model(model_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video '{source}'.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < max(8, int(fps * 0.6)):
        raise RuntimeError("The input video is too short for a reliable movement analysis.")

    dumbbell_mode = exercise_spec.tracking_mode == "dumbbell"
    height, width = frames[0].shape[:2]
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    source_start_frame = 0
    initial_plate: Optional[tuple[np.ndarray, float]] = None
    equipment_provenance = "unconfirmed"
    candidates_by_frame = _detect_candidates(frames, pose_model, min_pose_confidence)
    tracks = _build_person_tracks(candidates_by_frame, fps)
    preflight_compatibility = evaluate_barbell_press_preflight(tracks, fps)
    if not preflight_compatibility.compatible:
        raise ExerciseMismatchError(exercise_key, preflight_compatibility)
    dumbbell_tracking: Optional[tuple[list[Optional[np.ndarray]], list[float], list[float]]] = None
    dumbbell_side: Optional[str] = None
    dumbbell_lifter: Optional[PersonTrack] = None
    barbell_grip_tracking: Optional[tuple[list[Optional[np.ndarray]], list[float], list[float]]] = None
    barbell_grip_lifter: Optional[PersonTrack] = None
    if dumbbell_mode:
        dumbbell_path, dumbbell_confidence, dumbbell_radii, dumbbell_side, dumbbell_lifter = _track_dumbbell_handle(
            gray_frames,
            candidates_by_frame,
            near_arm,
        )
        dumbbell_tracking = (dumbbell_path, dumbbell_confidence, dumbbell_radii)
        equipment_provenance = "visual_dumbbell"
        initial_plate = (
            (dumbbell_path[0], dumbbell_radii[0])
            if _valid_point(dumbbell_path[0]) and dumbbell_radii[0] > 0.0
            else None
        )
    else:
        try:
            grip_lifter = _choose_lifter(tracks, height, fps)
        except RuntimeError:
            grip_lifter = None
        grip_path, grip_confidence, grip_radii, confirmed_grip_lifter = (
            _track_barbell_grip(gray_frames, grip_lifter, candidates_by_frame)
            if grip_lifter is not None
            else ([None] * len(frames), [0.0] * len(frames), [0.0] * len(frames), None)
        )
        plate_start_frame, plate_reference = _auto_trim_reference(gray_frames, fps)
        use_grip_reference = (
            _valid_point(grip_path[0])
            and grip_confidence[0] >= 0.70
            and _grip_track_is_reliable(grip_path, grip_confidence, fps)
            and plate_reference is None
        )
        if use_grip_reference:
            initial_plate = (grip_path[0], grip_radii[0])
            barbell_grip_tracking = (grip_path, grip_confidence, grip_radii)
            barbell_grip_lifter = confirmed_grip_lifter
            equipment_provenance = "pose_grip_fallback"
        else:
            source_start_frame, initial_plate = plate_start_frame, plate_reference
            if initial_plate is not None:
                equipment_provenance = "visual_plate"
            if source_start_frame:
                frames = frames[source_start_frame:]
                gray_frames = gray_frames[source_start_frame:]
                height, width = frames[0].shape[:2]
                candidates_by_frame = _detect_candidates(frames, pose_model, min_pose_confidence)
                tracks = _build_person_tracks(candidates_by_frame, fps)
                preflight_compatibility = evaluate_barbell_press_preflight(tracks, fps)
                if not preflight_compatibility.compatible:
                    raise ExerciseMismatchError(
                        exercise_key, preflight_compatibility
                    )
    if initial_plate is None and not dumbbell_mode:
        try:
            barbell_grip_lifter = _choose_lifter(tracks, height, fps)
        except RuntimeError:
            barbell_grip_lifter = None
        grip_path, grip_confidence, grip_radii, confirmed_grip_lifter = (
            _track_barbell_grip(gray_frames, barbell_grip_lifter, candidates_by_frame)
            if barbell_grip_lifter is not None
            else ([None] * len(frames), [0.0] * len(frames), [0.0] * len(frames), None)
        )
        if (
            _valid_point(grip_path[0])
            and grip_confidence[0] >= 0.70
            and _grip_track_is_reliable(grip_path, grip_confidence, fps)
        ):
            initial_plate = (grip_path[0], grip_radii[0])
            barbell_grip_tracking = (grip_path, grip_confidence, grip_radii)
            barbell_grip_lifter = confirmed_grip_lifter
            equipment_provenance = "pose_grip_fallback"
    if initial_plate is None:
        raise RuntimeError(
            "No reliable equipment reference was found in the first usable seconds. "
            "Keep the exercised hands and load visible at the start of the clip."
        )
    reference = FrameReference(0, initial_plate[0].copy(), float(initial_plate[1]))

    lifter: Optional[PersonTrack] = dumbbell_lifter or barbell_grip_lifter
    resolved_arm = near_arm or dumbbell_side or "unknown"
    anchor_shoulder: Optional[np.ndarray] = None
    anchor_hip: Optional[np.ndarray] = None
    shoulders: list[Optional[np.ndarray]] = [None] * len(frames)
    raw_elbows: list[Optional[np.ndarray]] = [None] * len(frames)
    raw_wrists: list[Optional[np.ndarray]] = [None] * len(frames)
    hips: list[Optional[np.ndarray]] = [None] * len(frames)
    landmark_quality: list[float] = [0.0] * len(frames)
    anatomy_candidate = False
    anatomy_reliable = False
    pose_quality_warning: Optional[str] = None

    plate_scale = (reference.plate_radius * 2.0) / 0.45
    calibration = PerspectiveCalibration(
        scale_px_per_m=plate_scale,
        yaw_radians=0.0,
        shoulder_width_px=float(levers["biacromial_width"]) * plate_scale,
        arm_length_px=float(levers["L_humerus"]) * plate_scale,
        facing_direction=1,
    )
    try:
        if lifter is None:
            lifter = _choose_lifter(tracks, height, fps)
        if resolved_arm not in {"left", "right"}:
            resolved_arm = _choose_near_arm(lifter, near_arm)
        shoulders, raw_elbows, raw_wrists, hips, landmark_quality = _paths_from_track(
            lifter, resolved_arm, len(frames)
        )
        pose_reference_index, candidate_shoulder, candidate_hip = _initial_pose_anchor(
            shoulders, raw_elbows, raw_wrists, hips, landmark_quality, fps
        )
        # Anatomy is rendered and scored only when it is genuinely present at
        # the same first-frame origin as the equipment reference.
        if pose_reference_index == reference.frame_index:
            anchor_shoulder, anchor_hip = candidate_shoulder, candidate_hip
            calibration = _calibrate_perspective(shoulders, hips, lifter, levers, landmark_quality)
            anatomy_candidate = True
        else:
            pose_quality_warning = (
                f"The selected body track starts at frame {pose_reference_index + 1}, "
                "so anatomy was excluded from the first-frame-anchored score."
            )
    except RuntimeError as error:
        pose_quality_warning = str(error)

    humerus_px = float(levers["L_humerus"]) * calibration.scale_px_per_m
    forearm_px = float(levers["L_forearm"]) * calibration.scale_px_per_m
    arm_reach_px = humerus_px + forearm_px
    roi = (0, 0, width, max(1, int(round(height * 0.72))))
    if dumbbell_tracking is not None:
        raw_bar_path, bar_confidence, plate_radii = dumbbell_tracking
    elif barbell_grip_tracking is not None:
        raw_bar_path, bar_confidence, plate_radii = barbell_grip_tracking
    else:
        raw_bar_path, bar_confidence, plate_radii = _track_equipment(
            gray_frames,
            roi,
            max(arm_reach_px, reference.plate_radius * 2.2),
            raw_wrists,
            anchor_shoulder,
            initial_detection=(reference.bar_center, reference.plate_radius),
        )
    shaft_support_by_track = {
        track.track_id: _bar_shaft_support_ratio(
            gray_frames, raw_bar_path, bar_confidence, plate_radii, track
        )
        for track in tracks
    }
    compatibility_track, compatibility = select_requested_exercise_track(
        exercise_spec.compatibility_policy,
        tracks,
        raw_bar_path,
        bar_confidence,
        plate_radii,
        reference.plate_radius,
        fps,
        equipment_provenance,
        shaft_support_by_track,
    )
    if compatibility_track is None or not compatibility.compatible:
        raise ExerciseMismatchError(exercise_key, compatibility)
    technique_warnings: list[str] = []
    if (
        compatibility.evidence.median_bar_to_shoulder_y is not None
        and compatibility.evidence.median_bar_to_shoulder_y <= 0.05
    ):
        technique_warnings.append(
            "The bar path may be below the preferred press zone; review depth "
            "and shoulder position."
        )
    _validate_equipment_track(raw_bar_path, bar_confidence, plate_radii, reference, fps)
    if lifter is not None and lifter.track_id != compatibility_track.track_id:
        anatomy_candidate = False
        pose_quality_warning = (
            "The equipment-associated person differs from the independently "
            "selected anatomy track, so limb scoring was suppressed."
        )
    bar_path = _smooth_points(
        _interpolate_short_gaps(raw_bar_path, max(2, int(fps * 0.20))),
        window=max(5, int(fps // 6) | 1),
    )
    # Smoothing must never move the immutable frame-one reference.
    bar_path[reference.frame_index] = reference.bar_center.copy()
    reps, y_values = _detect_reps(bar_path, bar_confidence, fps)
    if dumbbell_mode or barbell_grip_tracking is not None:
        reps = _filter_consistent_reps(reps, y_values)
    if not reps:
        raise RuntimeError(NO_COMPLETE_REPS_MESSAGE)
    series_end = reps[-1].end_frame
    for index in range(series_end + 1, len(bar_path)):
        bar_path[index] = None
        bar_confidence[index] = 0.0
        plate_radii[index] = 0.0

    if anatomy_candidate:
        pose_quality_warning = _anatomy_reliability_warning(
            shoulders,
            raw_elbows,
            raw_wrists,
            hips,
            landmark_quality,
            anchor_shoulder,
            float(levers["L_humerus"]) * calibration.scale_px_per_m,
            float(levers["L_forearm"]) * calibration.scale_px_per_m,
            reference.frame_index,
            series_end,
            fps,
        )
        anatomy_reliable = pose_quality_warning is None

    display_shoulders: list[Optional[np.ndarray]] = [None] * len(frames)
    display_elbows: list[Optional[np.ndarray]] = [None] * len(frames)
    elbow_observed_valid: list[bool] = [False] * len(frames)
    for index, (raw_shoulder, raw_elbow) in enumerate(zip(shoulders, raw_elbows)):
        if not anatomy_reliable:
            continue
        observed_ok = _valid_arm_chain(
            raw_shoulder,
            raw_elbow,
            raw_wrists[index],
            humerus_px,
            forearm_px,
            landmark_quality[index],
        )
        elbow_observed_valid[index] = observed_ok
        if observed_ok:
            display_shoulders[index] = raw_shoulder
            display_elbows[index] = raw_elbow

    if dumbbell_mode:
        bar_rep_scores, bar_frame_scores = _score_dumbbell_reps(reps, bar_path, bar_confidence)
        score_breakdown = ScoreBreakdown(
            bar_rep_scores=bar_rep_scores,
            bar_frame_scores=bar_frame_scores,
        )
        for rep_index, score in score_breakdown.bar_rep_scores.items():
            rep = reps[rep_index]
            rep.score = score

            # --- DODANA OCENA RAMIENIA DLA HANTLI ---
            limb_scores = []
            for frame_index in range(rep.start_frame, rep.end_frame + 1):
                elbow = display_elbows[frame_index]
                wrist = raw_wrists[frame_index]
                if elbow_observed_valid[frame_index] and _valid_point(elbow) and _valid_point(wrist):
                    forearm = wrist - elbow
                    f_len = float(np.linalg.norm(forearm))
                    if f_len > 8.0:
                        verticality_penalty = abs(float(forearm[0])) / f_len
                        limb_score = float(np.clip(100.0 - verticality_penalty * 34.0, 0.0, 100.0))
                        limb_scores.append(limb_score)
                        score_breakdown.limb_frame_scores[frame_index] = limb_score

            bar_frames_count = rep.end_frame - rep.start_frame + 1
            if len(limb_scores) >= max(2, int(math.ceil(bar_frames_count * 0.10))):
                l_score = int(round(float(np.mean(limb_scores))))
                score_breakdown.limb_rep_scores[rep_index] = l_score
                combined = int(round((score + l_score) / 2.0))
                score_breakdown.overall_rep_scores[rep_index] = combined
                rep.score = combined
            # ----------------------------------------
    else:
        score_breakdown = _score_reps(
            reps,
            bar_path,
            display_shoulders,
            display_elbows,
            raw_wrists,
            elbow_observed_valid,
            calibration,
            levers,
            trajectory_func,
            y_values,
            bar_confidence,
            reference,
        )
    completed_bar_scores = list(score_breakdown.bar_rep_scores.values())
    if not completed_bar_scores:
        raise RuntimeError("Complete repetitions were found, but none contained enough trustworthy tracking data to score.")
    # A motion cycle without a bar-path score is not presented as a completed
    # repetition. Limb scores remain optional and never downgrade into guesses.
    scored_rep_indices = sorted(score_breakdown.bar_rep_scores)
    reps = [reps[index] for index in scored_rep_indices]
    rep_score_rows = _aligned_rep_score_rows(score_breakdown)
    bar_path_score = int(round(float(np.mean(completed_bar_scores))))
    has_complete_limb_scores = (
        len(score_breakdown.limb_rep_scores) == len(score_breakdown.bar_rep_scores)
        and bool(score_breakdown.limb_rep_scores)
    )
    limb_motion_score = (
        int(round(float(np.mean(list(score_breakdown.limb_rep_scores.values())))))
        if has_complete_limb_scores
        else None
    )
    has_complete_overall_scores = (
        len(score_breakdown.overall_rep_scores) == len(score_breakdown.bar_rep_scores)
        and bool(score_breakdown.overall_rep_scores)
    )
    final_score = (
        int(round(float(np.mean(list(score_breakdown.overall_rep_scores.values())))))
        if has_complete_overall_scores
        else None
    )
    _render_output(
        frames,
        video_output_path,
        fps,
        exercise_key,
        resolved_arm,
        display_shoulders,
        reference,
        bar_path,
        bar_confidence,
        plate_radii,
        display_elbows,
        hips,
        reps,
        score_breakdown.bar_frame_scores,
        score_breakdown.limb_frame_scores,
        bar_path_score,
        limb_motion_score,
        final_score,
        "FULL ANATOMY" if anatomy_reliable else "BAR PATH ONLY",
    )

    summary = {
        "input_video": str(source),
        "source_start_frame": source_start_frame + 1,
        "output_video": str(video_output_path),
        "exercise": exercise_key,
        "exercise_compatibility": {
            **compatibility.to_dict(),
            "preflight": preflight_compatibility.to_dict(),
        },
        "compatibility_track_id": compatibility_track.track_id,
        "selected_track_id": lifter.track_id if lifter is not None else None,
        "near_arm": resolved_arm,
        "reference_frame": reference.frame_index + 1,
        "reference_bar_center": [round(float(value), 1) for value in reference.bar_center],
        "analysis_mode": "full_anatomy" if anatomy_reliable else "bar_path_only",
        "pose_quality_warning": pose_quality_warning,
        "technique_warnings": technique_warnings,
        "completed_reps": len(reps),
        "rep_scores": rep_score_rows,
        "bar_path_rep_scores": [row["bar_path_score"] for row in rep_score_rows],
        "limb_motion_rep_scores": [row["limb_motion_score"] for row in rep_score_rows],
        "overall_rep_scores": [row["final_score"] for row in rep_score_rows],
        "bar_path_score": bar_path_score,
        "limb_motion_score": limb_motion_score,
        "final_score": final_score,
        "equipment_tracking_coverage": round(float(sum(confidence >= 0.7 for confidence in bar_confidence)) / len(frames), 3),
        "camera_yaw_degrees": round(math.degrees(calibration.yaw_radians), 1),
    }
    return summary


def analyze_video_with_model(
    video_input_path: str | Path,
    video_output_path: str | Path = "vision_analysis.mp4",
    exercise_key: str = "Flat_Barbell_Press",
    *,
    model_path: Optional[str | Path] = None,
    near_arm: Optional[str] = None,
    min_pose_confidence: float = 0.35,
) -> dict:
    """
    Analyze a video with one conservative cross-platform retry.

    MediaPipe confidence boundaries can differ slightly across CPU delegates.
    A retry is allowed only when the otherwise valid analysis finds no complete
    repetitions; every identity, grip, anatomy, and equipment gate still runs.
    """
    thresholds = [float(min_pose_confidence)]
    configured_initial, configured_retry = RUNTIME_CONFIG.pose_confidence_attempts
    if math.isclose(min_pose_confidence, configured_initial, abs_tol=1e-9):
        thresholds.append(configured_retry)
    attempted: list[float] = []
    for threshold in thresholds:
        attempted.append(threshold)
        try:
            summary = _analyze_video_once(
                video_input_path,
                video_output_path,
                exercise_key,
                model_path=model_path,
                near_arm=near_arm,
                min_pose_confidence=threshold,
            )
            summary["pose_detection_confidence"] = round(threshold, 2)
            summary["pose_detection_attempts"] = [round(value, 2) for value in attempted]
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return summary
        except RuntimeError as error:
            if str(error) != NO_COMPLETE_REPS_MESSAGE or threshold == thresholds[-1]:
                raise
            print(
                f"No complete repetitions at pose confidence {threshold:.2f}; "
                f"retrying with {thresholds[len(attempted)]:.2f}."
            )
    raise RuntimeError(NO_COMPLETE_REPS_MESSAGE)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Layered barbell exercise-video analyzer")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", default="vision_analysis.mp4", help="Annotated output video path")
    parser.add_argument("--exercise", default="Flat_Barbell_Press", help="Exercise key from analyze_mid_chest.py")
    parser.add_argument("--near-arm", choices=("left", "right"), help="Optional manual camera-near arm override")
    parser.add_argument("--model", help="Optional MediaPipe .task model path")
    args = parser.parse_args()
    analyze_video_with_model(args.video, args.out, args.exercise, model_path=args.model, near_arm=args.near_arm)


if __name__ == "__main__":
    _main()