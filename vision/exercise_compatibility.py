"""Fail-closed requested-exercise compatibility checks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
NOSE = 0


@dataclass(frozen=True)
class CompatibilityEvidence:
    track_id: Optional[int]
    equipment_provenance: str
    direct_equipment_frames: int
    pose_supported_frames: int
    minimum_pose_supported_frames: int
    pose_support_ratio: float
    equipment_radius_observations: int
    minimum_radius_observations: int
    median_equipment_radius_ratio: Optional[float]
    equipment_vertical_span_pixels: Optional[float]
    minimum_vertical_span_pixels: float
    median_wrist_vertical_gap: Optional[float]
    median_wrist_span: Optional[float]
    median_plate_outside_grip_x: Optional[float]
    plate_outside_grip_ratio: float
    median_plate_radius_body_ratio: Optional[float]
    bar_shaft_support_ratio: float
    plate_grip_collinearity_ratio: float
    plate_wrist_distance_iqr: Optional[float]
    median_torso_angle_degrees: Optional[float]
    median_bar_to_shoulder_y: Optional[float]
    median_bar_to_hip_y: Optional[float]
    median_bar_to_nose_y: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityDecision:
    compatible: bool
    code: str
    reason: str
    evidence: object

    def to_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "code": self.code,
            "reason": self.reason,
            "evidence": getattr(self.evidence, "to_dict")(),
        }


class ExerciseMismatchError(RuntimeError):
    """Raised before repetition detection when the requested movement is absent."""

    def __init__(self, exercise_key: str, decision: CompatibilityDecision):
        self.exercise_key = exercise_key
        self.decision = decision
        super().__init__(
            f"Requested exercise mismatch [{decision.code}]: {decision.reason}"
        )


@dataclass(frozen=True)
class PosePreflightEvidence:
    minimum_two_hand_frames: int
    stable_two_hand_tracks: int
    median_torso_angles_degrees: tuple[float, ...]
    median_wrist_spans: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _valid_point(point: object) -> bool:
    if point is None:
        return False
    try:
        return bool(np.all(np.isfinite(np.asarray(point, dtype=float))))
    except (TypeError, ValueError):
        return False


def _median(rows: list[dict[str, Optional[float]]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else None


def collect_barbell_press_evidence(
    track: object,
    equipment_path: Sequence[object],
    equipment_confidence: Sequence[float],
    equipment_radii: Sequence[float],
    reference_radius: float,
    fps: float,
    equipment_provenance: str,
    bar_shaft_support_ratio: float = 0.0,
) -> CompatibilityEvidence:
    """Relate equipment observations to one continuous person track.

    X distance is intentionally not used: a tracked plate can be far from the
    grip midpoint, while its Y coordinate must still agree with the bar.
    """
    samples = list(getattr(track, "samples", []))
    candidate_by_frame = {
        int(getattr(sample, "frame_index")): getattr(sample, "candidate")
        for sample in samples
    }
    first_frame = min(candidate_by_frame, default=0)
    last_frame = max(candidate_by_frame, default=-1)
    direct_indices = [
        index
        for index, (point, quality) in enumerate(
            zip(equipment_path, equipment_confidence)
        )
        if (
            first_frame <= index <= last_frame
            and float(quality) >= 0.70
            and _valid_point(point)
        )
    ]
    rows: list[dict[str, Optional[float]]] = []
    for index in direct_indices:
        candidate = candidate_by_frame.get(index)
        if candidate is None:
            continue
        bar = np.asarray(equipment_path[index], dtype=float)
        points = getattr(candidate, "points", {})
        visibility = getattr(candidate, "visibility", {})
        required = (
            LEFT_WRIST,
            RIGHT_WRIST,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
        )
        if not all(
            landmark in points
            and float(visibility.get(landmark, 0.0)) >= 0.25
            and _valid_point(points[landmark])
            for landmark in required
        ):
            continue
        left_wrist = np.asarray(points[LEFT_WRIST], dtype=float)
        right_wrist = np.asarray(points[RIGHT_WRIST], dtype=float)
        grip = (left_wrist + right_wrist) / 2.0
        body_scale = max(24.0, float(getattr(candidate, "body_scale", 24.0)))
        vertical_gap = abs(float(grip[1] - bar[1])) / body_scale
        if vertical_gap > 0.80:
            continue
        shoulder = (
            np.asarray(points[LEFT_SHOULDER], dtype=float)
            + np.asarray(points[RIGHT_SHOULDER], dtype=float)
        ) / 2.0
        hip = (
            np.asarray(points[LEFT_HIP], dtype=float)
            + np.asarray(points[RIGHT_HIP], dtype=float)
        ) / 2.0
        torso = hip - shoulder
        torso_angle = math.degrees(
            math.atan2(abs(float(torso[1])), abs(float(torso[0])))
        )
        nose = points.get(NOSE)
        nose_visible = float(
            getattr(candidate, "visibility", {}).get(NOSE, 0.0)
        ) >= 0.25
        rows.append(
            {
                "wrist_gap": vertical_gap,
                "wrist_span": float(np.linalg.norm(left_wrist - right_wrist))
                / body_scale,
                "plate_outside_grip_x": (
                    (
                        min(float(left_wrist[0]), float(right_wrist[0]))
                        - float(bar[0])
                    )
                    / body_scale
                    if float(bar[0])
                    < min(float(left_wrist[0]), float(right_wrist[0]))
                    else (
                        (
                            float(bar[0])
                            - max(float(left_wrist[0]), float(right_wrist[0]))
                        )
                        / body_scale
                        if float(bar[0])
                        > max(float(left_wrist[0]), float(right_wrist[0]))
                        else -min(
                            abs(float(bar[0] - left_wrist[0])),
                            abs(float(bar[0] - right_wrist[0])),
                        )
                        / body_scale
                    )
                ),
                "plate_radius_body_ratio": (
                    float(equipment_radii[index]) / body_scale
                ),
                "plate_grip_collinearity": (
                    abs(
                        float(
                            (right_wrist[0] - left_wrist[0])
                            * (bar[1] - left_wrist[1])
                            - (right_wrist[1] - left_wrist[1])
                            * (bar[0] - left_wrist[0])
                        )
                    )
                    / max(1.0, float(np.linalg.norm(right_wrist - left_wrist)))
                    / body_scale
                ),
                "plate_left_wrist_distance": (
                    float(np.linalg.norm(bar - left_wrist)) / body_scale
                ),
                "plate_right_wrist_distance": (
                    float(np.linalg.norm(bar - right_wrist)) / body_scale
                ),
                "torso_angle": torso_angle,
                "bar_shoulder_y": float(bar[1] - shoulder[1]) / body_scale,
                "bar_hip_y": float(bar[1] - hip[1]) / body_scale,
                "bar_nose_y": (
                    float(bar[1] - np.asarray(nose, dtype=float)[1]) / body_scale
                    if nose_visible and _valid_point(nose)
                    else None
                ),
            }
        )
    positive_radii = [
        float(equipment_radii[index])
        for index in direct_indices
        if index < len(equipment_radii) and float(equipment_radii[index]) > 0.0
    ]
    reference_radius = max(1.0, float(reference_radius))
    direct_points = [
        np.asarray(equipment_path[index], dtype=float)
        for index in direct_indices
    ]
    vertical_span = (
        float(
            np.percentile(np.asarray(direct_points)[:, 1], 90)
            - np.percentile(np.asarray(direct_points)[:, 1], 10)
        )
        if direct_points
        else None
    )
    return CompatibilityEvidence(
        track_id=getattr(track, "track_id", None),
        equipment_provenance=equipment_provenance,
        direct_equipment_frames=len(direct_indices),
        pose_supported_frames=len(rows),
        minimum_pose_supported_frames=max(12, int(round(float(fps) * 1.35))),
        pose_support_ratio=len(rows) / max(1, len(direct_indices)),
        equipment_radius_observations=len(positive_radii),
        minimum_radius_observations=max(5, int(round(float(fps) * 0.30))),
        median_equipment_radius_ratio=(
            float(np.median(positive_radii)) / reference_radius
            if positive_radii
            else None
        ),
        equipment_vertical_span_pixels=vertical_span,
        minimum_vertical_span_pixels=max(14.0, reference_radius * 0.18),
        median_wrist_vertical_gap=_median(rows, "wrist_gap"),
        median_wrist_span=_median(rows, "wrist_span"),
        median_plate_outside_grip_x=_median(rows, "plate_outside_grip_x"),
        plate_outside_grip_ratio=(
            sum(
                float(row["plate_outside_grip_x"]) > 0.0
                for row in rows
            )
            / max(1, len(rows))
        ),
        median_plate_radius_body_ratio=_median(
            rows, "plate_radius_body_ratio"
        ),
        bar_shaft_support_ratio=float(bar_shaft_support_ratio),
        plate_grip_collinearity_ratio=(
            sum(
                float(row["plate_grip_collinearity"]) <= 0.15
                for row in rows
            )
            / max(1, len(rows))
        ),
        plate_wrist_distance_iqr=(
            max(
                float(
                    np.percentile(
                        [row["plate_left_wrist_distance"] for row in rows],
                        75,
                    )
                    - np.percentile(
                        [row["plate_left_wrist_distance"] for row in rows],
                        25,
                    )
                ),
                float(
                    np.percentile(
                        [row["plate_right_wrist_distance"] for row in rows],
                        75,
                    )
                    - np.percentile(
                        [row["plate_right_wrist_distance"] for row in rows],
                        25,
                    )
                ),
            )
            if rows
            else None
        ),
        median_torso_angle_degrees=_median(rows, "torso_angle"),
        median_bar_to_shoulder_y=_median(rows, "bar_shoulder_y"),
        median_bar_to_hip_y=_median(rows, "bar_hip_y"),
        median_bar_to_nose_y=_median(rows, "bar_nose_y"),
    )


def evaluate_barbell_press_preflight(
    tracks: Sequence[object],
    fps: float,
) -> CompatibilityDecision:
    """Reject definite pose mismatches before expensive equipment detection."""
    minimum_frames = max(8, int(round(float(fps) * 0.40)))
    track_profiles: list[tuple[float, float]] = []
    required = (
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    )
    for track in tracks:
        angles: list[float] = []
        wrist_spans: list[float] = []
        for sample in getattr(track, "samples", []):
            candidate = getattr(sample, "candidate", None)
            if candidate is None:
                continue
            points = getattr(candidate, "points", {})
            visibility = getattr(candidate, "visibility", {})
            if not all(
                landmark in points
                and float(visibility.get(landmark, 0.0)) >= 0.35
                and _valid_point(points[landmark])
                for landmark in required
            ):
                continue
            shoulder = (
                np.asarray(points[LEFT_SHOULDER], dtype=float)
                + np.asarray(points[RIGHT_SHOULDER], dtype=float)
            ) / 2.0
            hip = (
                np.asarray(points[LEFT_HIP], dtype=float)
                + np.asarray(points[RIGHT_HIP], dtype=float)
            ) / 2.0
            torso = hip - shoulder
            body_scale = max(24.0, float(getattr(candidate, "body_scale", 24.0)))
            angles.append(
                math.degrees(
                    math.atan2(abs(float(torso[1])), abs(float(torso[0])))
                )
            )
            wrist_spans.append(
                float(
                    np.linalg.norm(
                        np.asarray(points[LEFT_WRIST], dtype=float)
                        - np.asarray(points[RIGHT_WRIST], dtype=float)
                    )
                )
                / body_scale
            )
        if len(angles) >= minimum_frames:
            track_profiles.append(
                (float(np.median(angles)), float(np.median(wrist_spans)))
            )
    evidence = PosePreflightEvidence(
        minimum_two_hand_frames=minimum_frames,
        stable_two_hand_tracks=len(track_profiles),
        median_torso_angles_degrees=tuple(row[0] for row in track_profiles),
        median_wrist_spans=tuple(row[1] for row in track_profiles),
    )
    if not track_profiles:
        return CompatibilityDecision(
            False,
            "no_two_hand_setup",
            "No continuous person track shows both hands and the torso long enough "
            "to confirm a flat barbell press.",
            evidence,
        )
    wide_grip_profiles = [
        profile for profile in track_profiles if profile[1] >= 0.30
    ]
    if not wide_grip_profiles:
        return CompatibilityDecision(
            False,
            "narrow_grip_pattern",
            "Every stable two-hand track uses a grip too narrow for a flat "
            "barbell press.",
            evidence,
        )
    if min(profile[0] for profile in wide_grip_profiles) > 80.0:
        return CompatibilityDecision(
            False,
            "upright_press_pattern",
            "Every stable two-hand person track remains upright, which is "
            "incompatible with a flat bench press.",
            evidence,
        )
    return CompatibilityDecision(
        True,
        "preflight_compatible",
        "A stable two-hand, non-upright press setup is present.",
        evidence,
    )


def evaluate_flat_barbell_press(
    evidence: CompatibilityEvidence,
) -> CompatibilityDecision:
    """Require independent equipment, grip, and posture evidence."""
    checks = (
        (
            evidence.direct_equipment_frames
            >= evidence.minimum_pose_supported_frames,
            "insufficient_equipment",
            "Too few direct equipment observations confirm a barbell movement.",
        ),
        (
            evidence.equipment_provenance == "visual_plate",
            "unconfirmed_barbell_equipment",
            "Hand motion alone cannot confirm that a physical barbell is present.",
        ),
        (
            evidence.pose_supported_frames
            >= evidence.minimum_pose_supported_frames,
            "equipment_pose_disconnected",
            "Too few observations connect the tracked load to one continuous lifter.",
        ),
        (
            evidence.pose_support_ratio >= 0.22,
            "equipment_pose_disconnected",
            "The tracked load does not stay vertically aligned with two visible hands.",
        ),
        (
            evidence.plate_outside_grip_ratio >= 0.50
            and (
                (
                    evidence.median_plate_outside_grip_x is not None
                    and evidence.median_plate_outside_grip_x >= 0.70
                )
                or (
                    evidence.bar_shaft_support_ratio >= 0.25
                    and (
                        (
                            evidence.median_plate_outside_grip_x is not None
                            and evidence.median_plate_outside_grip_x >= 0.20
                        )
                        or (
                            evidence.median_plate_radius_body_ratio is not None
                            and evidence.median_plate_radius_body_ratio >= 0.30
                        )
                    )
                )
            ),
            "non_barbell_load_geometry",
            "The visible round load does not show a long barbell lever or a "
            "repeated shared shaft through the two-hand grip.",
        ),
        (
            evidence.equipment_radius_observations
            >= evidence.minimum_radius_observations,
            "unconfirmed_barbell_equipment",
            "The tracked object is not repeatedly confirmed as a barbell plate or grip.",
        ),
        (
            evidence.median_equipment_radius_ratio is not None
            and 0.72 <= evidence.median_equipment_radius_ratio <= 1.48,
            "unstable_equipment_scale",
            "The tracked object changes scale too much to be the frame-one barbell.",
        ),
        (
            evidence.equipment_vertical_span_pixels is not None
            and evidence.equipment_vertical_span_pixels
            >= evidence.minimum_vertical_span_pixels,
            "static_equipment",
            "The tracked object remains effectively static instead of following a press.",
        ),
        (
            evidence.median_wrist_vertical_gap is not None
            and evidence.median_wrist_vertical_gap <= 0.65,
            "equipment_pose_disconnected",
            "The tracked load is not consistently aligned with the two-hand grip.",
        ),
        (
            evidence.median_wrist_span is not None
            and evidence.median_wrist_span >= 0.30,
            "narrow_grip_pattern",
            "The two-hand grip is too narrow for a flat barbell press.",
        ),
        (
            evidence.median_torso_angle_degrees is not None
            and evidence.median_torso_angle_degrees <= 80.0,
            "upright_press_pattern",
            "The lifter remains upright, which is incompatible with a flat bench press.",
        ),
        (
            evidence.median_bar_to_shoulder_y is not None
            and evidence.median_bar_to_shoulder_y <= 0.05,
            "load_below_press_zone",
            "The tracked load remains below the shoulder press zone.",
        ),
    )
    for accepted, code, reason in checks:
        if not accepted:
            return CompatibilityDecision(False, code, reason, evidence)
    return CompatibilityDecision(
        True,
        "compatible",
        "Equipment, two-hand grip, and body orientation match a flat barbell press.",
        evidence,
    )


def evaluate_requested_exercise(
    compatibility_policy: str,
    evidence: CompatibilityEvidence,
) -> CompatibilityDecision:
    if compatibility_policy == "flat_barbell_press":
        return evaluate_flat_barbell_press(evidence)
    raise ValueError(
        f"No compatibility evaluator is registered for '{compatibility_policy}'."
    )


def select_requested_exercise_track(
    compatibility_policy: str,
    tracks: Sequence[object],
    equipment_path: Sequence[object],
    equipment_confidence: Sequence[float],
    equipment_radii: Sequence[float],
    reference_radius: float,
    fps: float,
    equipment_provenance: str,
    bar_shaft_support_by_track: Optional[dict[int, float]] = None,
) -> tuple[Optional[object], CompatibilityDecision]:
    """Select one equipment-associated track; never combine people per frame."""
    accepted: list[tuple[CompatibilityEvidence, object, CompatibilityDecision]] = []
    rejected: list[tuple[CompatibilityEvidence, CompatibilityDecision]] = []
    preflight_rejections: list[CompatibilityDecision] = []
    for track in tracks:
        preflight = evaluate_barbell_press_preflight([track], fps)
        if not preflight.compatible:
            preflight_rejections.append(preflight)
            continue
        evidence = collect_barbell_press_evidence(
            track,
            equipment_path,
            equipment_confidence,
            equipment_radii,
            reference_radius,
            fps,
            equipment_provenance,
            (
                (bar_shaft_support_by_track or {}).get(
                    getattr(track, "track_id", -1), 0.0
                )
            ),
        )
        decision = evaluate_requested_exercise(compatibility_policy, evidence)
        if decision.compatible:
            accepted.append((evidence, track, decision))
        else:
            rejected.append((evidence, decision))
    if accepted:
        _, track, decision = max(
            accepted,
            key=lambda item: (
                item[0].pose_supported_frames,
                item[0].pose_support_ratio,
            ),
        )
        return track, decision
    if rejected:
        _, decision = max(
            rejected,
            key=lambda item: (
                item[0].pose_supported_frames,
                item[0].pose_support_ratio,
            ),
        )
        return None, decision
    if preflight_rejections:
        decision = max(
            preflight_rejections,
            key=lambda item: getattr(item.evidence, "stable_two_hand_tracks", 0),
        )
        return None, decision
    return None, evaluate_barbell_press_preflight([], fps)