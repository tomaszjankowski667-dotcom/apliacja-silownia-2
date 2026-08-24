import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vision.exercise_compatibility import (
    CompatibilityEvidence,
    collect_barbell_press_evidence,
    collect_dumbbell_press_evidence,
    evaluate_barbell_press_preflight,
    evaluate_flat_barbell_press,
    evaluate_flat_dumbbell_press,
    select_requested_exercise_track,
)
from vision.exercise_registry import get_exercise_spec


@dataclass
class Candidate:
    points: dict[int, np.ndarray]
    visibility: dict[int, float]
    body_scale: float = 100.0


def make_candidate(torso_angle_degrees: float, wrist_y: float = 120.0) -> Candidate:
    angle = np.radians(torso_angle_degrees)
    shoulder = np.array([200.0, 220.0])
    hip = shoulder + np.array([100.0 * np.cos(angle), 100.0 * np.sin(angle)])
    points = {
        0: np.array([200.0, 180.0]),
        11: shoulder + np.array([-20.0, 0.0]),
        12: shoulder + np.array([20.0, 0.0]),
        15: np.array([165.0, wrist_y]),
        16: np.array([235.0, wrist_y]),
        23: hip + np.array([-20.0, 0.0]),
        24: hip + np.array([20.0, 0.0]),
    }
    return Candidate(points, {index: 1.0 for index in points})


def make_track(
    candidate: Candidate,
    frame_count: int = 30,
    *,
    track_id: int = 1,
    start_frame: int = 0,
):
    return SimpleNamespace(
        track_id=track_id,
        samples=[
            SimpleNamespace(
                frame_index=start_frame + index,
                candidate=candidate,
            )
            for index in range(frame_count)
        ],
    )


def collect(candidate: Candidate, path: list[np.ndarray]):
    return collect_barbell_press_evidence(
        make_track(candidate, len(path)),
        path,
        [1.0] * len(path),
        [10.0] * len(path),
        10.0,
        30.0,
        "visual_plate",
        1.0,
    )


def moving_path() -> list[np.ndarray]:
    return [
        np.array([300.0, y])
        for y in np.linspace(110.0, 130.0, 45)
    ]


class ExerciseCompatibilityTests(unittest.TestCase):
    def test_preflight_rejects_missing_two_hand_track(self):
        candidate = make_candidate(20.0)
        candidate.visibility[16] = 0.0
        track = make_track(candidate)

        decision = evaluate_barbell_press_preflight([track], 30.0)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "no_two_hand_setup")

    def test_preflight_rejects_upright_two_hand_track(self):
        candidate = make_candidate(88.0)
        track = make_track(candidate)

        decision = evaluate_barbell_press_preflight([track], 30.0)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "upright_press_pattern")

    def test_preflight_rejects_narrow_cable_grip(self):
        candidate = make_candidate(40.0)
        candidate.points[15] = np.array([197.0, 120.0])
        candidate.points[16] = np.array([203.0, 120.0])
        track = make_track(candidate)

        decision = evaluate_barbell_press_preflight([track], 30.0)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "narrow_grip_pattern")

    def test_side_view_flat_press_is_accepted(self):
        candidate = make_candidate(20.0)
        path = moving_path()
        evidence = collect(candidate, path)

        decision = evaluate_flat_barbell_press(evidence)

        self.assertTrue(decision.compatible)
        self.assertEqual(decision.code, "compatible")

    def test_head_on_flat_press_projection_is_accepted(self):
        candidate = make_candidate(74.0)
        path = moving_path()
        evidence = collect(candidate, path)

        self.assertTrue(evaluate_flat_barbell_press(evidence).compatible)

    def test_hand_associated_dumbbell_press_is_accepted(self):
        candidate = make_candidate(20.0)
        path = [
            np.array([165.0, y])
            for y in np.linspace(110.0, 130.0, 45)
        ]
        evidence = collect_dumbbell_press_evidence(
            make_track(candidate, len(path)),
            path,
            [1.0] * len(path),
            30.0,
        )
        decision = evaluate_flat_dumbbell_press(evidence)

        self.assertTrue(get_exercise_spec("Flat_Dumbbell_Press").video_supported)
        self.assertTrue(decision.compatible)
        self.assertEqual(decision.code, "dumbbell_press_compatible")

    def test_upright_overhead_press_is_rejected(self):
        candidate = make_candidate(88.0)
        path = moving_path()
        evidence = collect(candidate, path)

        decision = evaluate_flat_barbell_press(evidence)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "upright_press_pattern")

    def test_disconnected_equipment_is_rejected(self):
        candidate = make_candidate(20.0, wrist_y=300.0)
        path = [np.array([300.0, 100.0]) for _ in range(45)]
        evidence = collect(candidate, path)

        decision = evaluate_flat_barbell_press(evidence)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "equipment_pose_disconnected")

    def test_short_tracks_cannot_be_combined_across_people(self):
        candidate = make_candidate(20.0)
        tracks = [
            make_track(candidate, 15, track_id=1, start_frame=0),
            make_track(candidate, 15, track_id=2, start_frame=15),
        ]
        path = [np.array([300.0, 118.0]) for _ in range(30)]

        track, decision = select_requested_exercise_track(
            "flat_barbell_press",
            tracks,
            path,
            [1.0] * len(path),
            [10.0] * len(path),
            10.0,
            30.0,
            "visual_plate",
        )

        self.assertIsNone(track)
        self.assertFalse(decision.compatible)

    def test_pose_grip_fallback_cannot_confirm_a_barbell(self):
        candidate = make_candidate(20.0)
        path = moving_path()
        evidence = collect_barbell_press_evidence(
            make_track(candidate, len(path)),
            path,
            [1.0] * len(path),
            [10.0] * len(path),
            10.0,
            30.0,
            "pose_grip_fallback",
        )

        decision = evaluate_flat_barbell_press(evidence)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "unconfirmed_barbell_equipment")

    def test_round_load_near_one_hand_cannot_confirm_a_barbell(self):
        candidate = make_candidate(20.0)
        path = [
            np.array([240.0, y])
            for y in np.linspace(110.0, 130.0, 45)
        ]
        evidence = collect(candidate, path)

        decision = evaluate_flat_barbell_press(evidence)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "guided_or_machine_load_geometry")

    def test_uncertain_projection_track_is_returned_only_for_explicit_override(self):
        candidate = make_candidate(20.0)
        path = [
            np.array([270.0, y])
            for y in np.linspace(110.0, 130.0, 45)
        ]
        args = (
            "flat_barbell_press",
            [make_track(candidate, len(path))],
            path,
            [1.0] * len(path),
            [35.0] * len(path),
            35.0,
            30.0,
            "visual_plate",
            {1: 0.0},
        )

        strict_track, strict_decision = select_requested_exercise_track(*args)
        override_track, override_decision = select_requested_exercise_track(
            *args,
            return_best_rejected_track=True,
        )

        self.assertIsNone(strict_track)
        self.assertEqual(strict_decision.code, "non_barbell_load_geometry")
        self.assertIsNotNone(override_track)
        self.assertEqual(
            override_decision.code,
            "non_barbell_load_geometry",
        )

    def test_large_dumbbell_plate_without_shared_shaft_is_rejected(self):
        candidate = make_candidate(20.0)
        path = [
            np.array([270.0, y])
            for y in np.linspace(110.0, 130.0, 45)
        ]
        evidence = collect_barbell_press_evidence(
            make_track(candidate, len(path)),
            path,
            [1.0] * len(path),
            [35.0] * len(path),
            35.0,
            30.0,
            "visual_plate",
            0.0,
        )

        decision = evaluate_flat_barbell_press(evidence)

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.code, "non_barbell_load_geometry")

    def test_unsupported_template_is_rejected_before_video_analysis(self):
        with self.assertRaisesRegex(ValueError, "not yet supported"):
            get_exercise_spec("Chest_Dips")

    def test_manifest_covers_every_source_recording_once(self):
        manifest_path = Path(__file__).with_name("video_corpus_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = [row["file"] for row in manifest["recordings"]]

        self.assertEqual(len(names), 20)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(names),
            {"k_1787495418440.mp4"}
            | {"seria_1787581108845.mp4"}
            | {
                next(
                    Path("attached_assets").glob(f"k{index}_*.mp4")
                ).name
                for index in range(1, 19)
            },
        )


if __name__ == "__main__":
    unittest.main()