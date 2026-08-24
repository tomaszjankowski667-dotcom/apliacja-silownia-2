import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision.vision_exercise_analyzer import (
    FrameReference,
    LEFT_WRIST,
    RIGHT_WRIST,
    PoseCandidate,
    PersonTrack,
    PerspectiveCalibration,
    ScoreBreakdown,
    NO_COMPLETE_REPS_MESSAGE,
    Rep,
    _detect_reps,
    _filter_consistent_reps,
    _anatomy_reliability_warning,
    _aligned_rep_score_rows,
    _can_override_equipment_compatibility,
    _grip_track_is_reliable,
    _press_zone_depth_penalty,
    _track_barbell_grip,
    _score_dumbbell_reps,
    _score_reps,
    _select_reference_plate,
    _track_equipment,
    _validate_equipment_track,
    analyze_video_with_model,
)


class FirstFrameReferenceTests(unittest.TestCase):
    def test_tracker_keeps_moving_plate_when_static_circle_is_nearby(self):
        frame_count = 48
        height, width = 360, 480
        frames = []
        expected = []
        for index in range(frame_count):
            frame = np.full((height, width), 210, dtype=np.uint8)
            center = np.array([290.0 + index * 0.35, 105.0 + 42.0 * np.sin(index / 8.0)])
            expected.append(center)
            cv2.circle(frame, tuple(center.astype(int)), 48, 35, 8)
            cv2.circle(frame, tuple(center.astype(int)), 19, 90, 5)
            cv2.circle(frame, (170, 155), 50, 45, 8)
            frames.append(frame)

        reference = (expected[0], 48.0)
        path, confidence, radii = _track_equipment(
            frames,
            (0, 0, width, height),
            150.0,
            [None] * frame_count,
            None,
            reference,
        )

        tracked = np.asarray(path, dtype=float)
        expected_points = np.asarray(expected, dtype=float)
        self.assertGreaterEqual(sum(value >= 0.70 for value in confidence), int(frame_count * 0.60))
        self.assertLess(float(np.median(np.linalg.norm(tracked - expected_points, axis=1))), 12.0)
        _validate_equipment_track(
            path,
            confidence,
            radii,
            FrameReference(0, expected[0], 48.0),
            30.0,
        )

    def test_reference_selection_prefers_dominant_vertical_motion(self):
        frame_count = 54
        height, width = 360, 480
        frames = []
        target_start = np.array([310.0, 92.0])
        for index in range(frame_count):
            frame = np.full((height, width), 210, dtype=np.uint8)
            target = target_start + np.array([index * 0.20, index * 1.55])
            distractor = np.array([155.0 + index * 0.15, 145.0 + 8.0 * np.sin(index / 7.0)])
            cv2.circle(frame, tuple(target.astype(int)), 47, 35, 8)
            cv2.circle(frame, tuple(target.astype(int)), 18, 90, 5)
            cv2.circle(frame, tuple(distractor.astype(int)), 55, 35, 8)
            cv2.circle(frame, tuple(distractor.astype(int)), 20, 90, 5)
            frames.append(frame)

        selected = _select_reference_plate(frames, 30.0)

        self.assertIsNotNone(selected)
        self.assertLess(float(np.linalg.norm(selected[0] - target_start)), 18.0)

    def test_static_background_circle_is_rejected(self):
        point = np.array([240.0, 120.0])
        path = [point.copy() for _ in range(40)]
        confidence = [1.0] * len(path)
        radii = [50.0] * len(path)
        with self.assertRaisesRegex(RuntimeError, "static"):
            _validate_equipment_track(
                path,
                confidence,
                radii,
                FrameReference(0, point, 50.0),
                30.0,
            )

    def test_broad_lockout_plateau_splits_two_repetitions(self):
        first_down = np.linspace(250.0, 370.0, 50)
        first_up = np.linspace(370.0, 260.0, 48)
        plateau = np.full(20, 260.0)
        second_down = np.linspace(260.0, 355.0, 42)
        second_up = np.linspace(355.0, 260.0, 59)
        y_values = np.concatenate([first_down, first_up, plateau, second_down, second_up])
        path = [np.array([320.0, value]) for value in y_values]
        confidence = [1.0] * len(path)

        reps, _ = _detect_reps(path, confidence, 30.0)

        self.assertEqual(len(reps), 2)
        self.assertLess(reps[0].end_frame, reps[1].bottom_frame)
        self.assertIsInstance(reps[0], Rep)

    def test_small_post_set_cycle_is_not_counted_as_a_rep(self):
        full_rep = np.concatenate([
            np.linspace(250.0, 370.0, 42),
            np.linspace(370.0, 250.0, 42)[1:],
        ])
        post_set_motion = np.concatenate([
            np.linspace(250.0, 310.0, 24),
            np.linspace(310.0, 250.0, 24)[1:],
        ])
        y_values = np.concatenate([np.full(12, 250.0), full_rep, full_rep[1:], full_rep[1:], post_set_motion[1:]])
        path = [np.array([320.0, value]) for value in y_values]
        confidence = [1.0] * len(path)

        reps, detected_y = _detect_reps(path, confidence, 30.0)
        reps = _filter_consistent_reps(reps, detected_y)

        self.assertEqual(len(reps), 3)
        self.assertTrue(all(rep.rom_px > 100.0 for rep in reps))

    def test_shifted_lockout_after_standing_is_not_counted(self):
        detected_y = np.full(320, 250.0)
        reps = []
        for index, top in enumerate((250.0, 250.0, 250.0, 180.0)):
            start = index * 75
            bottom = start + 35
            end = start + 70
            detected_y[start] = top
            detected_y[bottom] = top + 120.0
            detected_y[end] = top
            reps.append(Rep(start, bottom, end, 120.0))

        reps = _filter_consistent_reps(reps, detected_y)

        self.assertEqual(len(reps), 3)
        self.assertTrue(all(abs(detected_y[rep.start_frame] - 250.0) < 15.0 for rep in reps))

    def test_repeatable_dumbbell_paths_receive_a_strong_path_score(self):
        reps = []
        path = []
        confidence = []
        frame = 0
        for lateral_offset in (0.0, 6.0, -5.0):
            down_y = np.linspace(250.0, 370.0, 36)
            up_y = np.linspace(370.0, 250.0, 36)[1:]
            y_values = np.concatenate([down_y, up_y])
            progress = np.concatenate([
                np.linspace(0.0, 1.0, len(down_y)),
                np.linspace(1.0, 0.0, len(up_y)),
            ])
            x_values = 320.0 + 28.0 * progress + lateral_offset * np.sin(np.pi * progress)
            start = frame
            bottom = frame + len(down_y) - 1
            end = frame + len(y_values) - 1
            reps.append(Rep(start, bottom, end, 120.0))
            path.extend(np.array([x, y]) for x, y in zip(x_values, y_values))
            confidence.extend([1.0] * len(y_values))
            frame = end + 1

        scores, _ = _score_dumbbell_reps(reps, path, confidence)

        self.assertEqual(len(scores), 3)
        self.assertGreaterEqual(min(scores.values()), 80)
        self.assertLessEqual(max(scores.values()), 100)

    def test_single_dumbbell_rep_does_not_create_its_own_perfect_template(self):
        y_values = np.concatenate([np.linspace(250.0, 370.0, 36), np.linspace(370.0, 250.0, 36)[1:]])
        x_values = np.linspace(300.0, 390.0, len(y_values))
        path = [np.array([x, y]) for x, y in zip(x_values, y_values)]
        confidence = [1.0] * len(path)
        rep = Rep(0, 35, len(path) - 1, 120.0)

        scores, frame_scores = _score_dumbbell_reps([rep], path, confidence)

        self.assertEqual(scores, {})
        self.assertEqual(frame_scores, {})

    def test_bar_path_and_limb_motion_receive_independent_scores(self):
        down_y = np.linspace(250.0, 370.0, 36)
        up_y = np.linspace(370.0, 250.0, 36)[1:]
        y_values = np.concatenate([down_y, up_y])
        path = [np.array([320.0, y]) for y in y_values]
        elbows = [np.array([320.0, y + 75.0]) for y in y_values]
        shoulders = [np.array([285.0, y + 35.0]) for y in y_values]
        rep = Rep(0, len(down_y) - 1, len(y_values) - 1, 120.0)
        calibration = PerspectiveCalibration(100.0, 0.0, 41.0, 100.0, 1)

        result = _score_reps(
            [rep],
            path,
            shoulders,
            elbows,
            path,
            [True] * len(path),
            calibration,
            {},
            lambda progress, levers, phase: np.array([0.0, 0.0, 0.0]),
            y_values,
            [1.0] * len(path),
            FrameReference(0, path[0], 45.0),
        )

        self.assertGreaterEqual(result.bar_rep_scores[0], 95)
        self.assertGreaterEqual(result.limb_rep_scores[0], 95)
        self.assertGreaterEqual(result.overall_rep_scores[0], 95)

        low_depth_result = _score_reps(
            [rep],
            path,
            shoulders,
            elbows,
            path,
            [True] * len(path),
            calibration,
            {},
            lambda progress, levers, phase: np.array([0.0, 0.0, 0.0]),
            y_values,
            [1.0] * len(path),
            FrameReference(0, path[0], 45.0),
            0.85,
        )

        self.assertGreaterEqual(
            result.bar_rep_scores[0] - low_depth_result.bar_rep_scores[0],
            15,
        )

    def test_press_zone_depth_penalty_uses_downward_image_y(self):
        self.assertEqual(_press_zone_depth_penalty(-0.40), 0.0)
        self.assertEqual(_press_zone_depth_penalty(0.05), 0.0)
        self.assertGreater(_press_zone_depth_penalty(0.80), 0.0)

    def test_uncertainty_override_cannot_bypass_hard_equipment_mismatches(self):
        self.assertTrue(
            _can_override_equipment_compatibility(
                "uncertain_axial_barbell_projection"
            )
        )
        self.assertTrue(
            _can_override_equipment_compatibility(
                "uncertain_barbell_lifter_association"
            )
        )
        self.assertFalse(
            _can_override_equipment_compatibility(
                "non_barbell_load_geometry"
            )
        )
        self.assertFalse(
            _can_override_equipment_compatibility(
                "guided_or_machine_load_geometry"
            )
        )

    def test_missing_limb_observations_do_not_hide_a_valid_bar_score(self):
        down_y = np.linspace(250.0, 370.0, 36)
        up_y = np.linspace(370.0, 250.0, 36)[1:]
        y_values = np.concatenate([down_y, up_y])
        path = [np.array([320.0, y]) for y in y_values]
        rep = Rep(0, len(down_y) - 1, len(y_values) - 1, 120.0)

        result = _score_reps(
            [rep],
            path,
            [None] * len(path),
            [None] * len(path),
            [None] * len(path),
            [False] * len(path),
            PerspectiveCalibration(100.0, 0.0, 41.0, 100.0, 1),
            {},
            lambda progress, levers, phase: np.array([0.0, 0.0, 0.0]),
            y_values,
            [1.0] * len(path),
            FrameReference(0, path[0], 45.0),
        )

        self.assertGreaterEqual(result.bar_rep_scores[0], 95)
        self.assertEqual(result.limb_rep_scores, {})
        self.assertEqual(result.overall_rep_scores, {})

    def test_partial_limb_scores_stay_aligned_with_their_repetition(self):
        rows = _aligned_rep_score_rows(ScoreBreakdown(
            bar_rep_scores={0: 82, 1: 76},
            limb_rep_scores={1: 91},
            overall_rep_scores={1: 84},
        ))

        self.assertEqual(rows, [
            {
                "detected_rep_number": 1,
                "bar_path_score": 82,
                "limb_motion_score": None,
                "final_score": None,
            },
            {
                "detected_rep_number": 2,
                "bar_path_score": 76,
                "limb_motion_score": 91,
                "final_score": 84,
            },
        ])

    def test_filtered_middle_rep_keeps_original_rep_numbers(self):
        rows = _aligned_rep_score_rows(ScoreBreakdown(
            bar_rep_scores={0: 88, 2: 79},
            limb_rep_scores={0: 90, 2: 85},
            overall_rep_scores={0: 89, 2: 82},
        ))

        self.assertEqual([row["detected_rep_number"] for row in rows], [1, 3])
        self.assertEqual([row["bar_path_score"] for row in rows], [88, 79])

    @patch("vision.vision_exercise_analyzer._analyze_video_once")
    def test_no_rep_failure_retries_at_cross_platform_pose_threshold(self, analyze_once):
        analyze_once.side_effect = [
            RuntimeError(NO_COMPLETE_REPS_MESSAGE),
            {"completed_reps": 5},
        ]

        result = analyze_video_with_model("input.mp4", "output.mp4")

        self.assertEqual(analyze_once.call_count, 2)
        self.assertEqual(analyze_once.call_args_list[0].kwargs["min_pose_confidence"], 0.35)
        self.assertEqual(analyze_once.call_args_list[1].kwargs["min_pose_confidence"], 0.30)
        self.assertEqual(result["pose_detection_confidence"], 0.30)
        self.assertEqual(result["pose_detection_attempts"], [0.35, 0.30])

    @patch("vision.vision_exercise_analyzer._analyze_video_once")
    def test_unrelated_runtime_error_does_not_retry(self, analyze_once):
        analyze_once.side_effect = RuntimeError("No reliable equipment reference")

        with self.assertRaisesRegex(RuntimeError, "No reliable equipment reference"):
            analyze_video_with_model("input.mp4", "output.mp4")

        analyze_once.assert_called_once()

    def test_anatomy_gate_accepts_continuous_visible_joint_data(self):
        frame_count = 90
        shoulder = np.array([320.0, 230.0])
        shoulders = [shoulder.copy() for _ in range(frame_count)]
        elbows = [np.array([355.0, 315.0]) for _ in range(frame_count)]
        wrists = [np.array([385.0, 400.0]) for _ in range(frame_count)]
        hips = [np.array([315.0, 500.0]) for _ in range(frame_count)]
        quality = [0.82] * frame_count

        warning = _anatomy_reliability_warning(
            shoulders, elbows, wrists, hips, quality, shoulder, 92.0, 92.0, 0, frame_count - 1, 30.0
        )

        self.assertIsNone(warning)

    def test_anatomy_gate_rejects_fragmented_joint_data(self):
        frame_count = 90
        shoulder = np.array([320.0, 230.0])
        shoulders = [shoulder.copy() if index < 20 or index > 70 else None for index in range(frame_count)]
        elbows = [np.array([355.0, 315.0]) if point is not None else None for point in shoulders]
        wrists = [np.array([385.0, 400.0]) if point is not None else None for point in shoulders]
        hips = [np.array([315.0, 500.0]) if point is not None else None for point in shoulders]
        quality = [0.82 if point is not None else 0.0 for point in shoulders]

        warning = _anatomy_reliability_warning(
            shoulders, elbows, wrists, hips, quality, shoulder, 92.0, 92.0, 0, frame_count - 1, 30.0
        )

        self.assertIsNotNone(warning)

    def test_anatomy_gate_requires_complete_frame_one_arm_chain(self):
        frame_count = 90
        shoulder = np.array([320.0, 230.0])
        shoulders = [shoulder.copy() for _ in range(frame_count)]
        elbows = [np.array([355.0, 315.0]) for _ in range(frame_count)]
        wrists = [np.array([385.0, 400.0]) for _ in range(frame_count)]
        hips = [np.array([315.0, 500.0]) for _ in range(frame_count)]
        quality = [0.82] * frame_count
        wrists[0] = None
        quality[0] = 0.0

        warning = _anatomy_reliability_warning(
            shoulders, elbows, wrists, hips, quality, shoulder, 92.0, 92.0, 0, frame_count - 1, 30.0
        )

        self.assertIn("Frame 1", warning)

    def test_anatomy_gate_rejects_impossible_wrist_distance(self):
        frame_count = 90
        shoulder = np.array([320.0, 230.0])
        shoulders = [shoulder.copy() for _ in range(frame_count)]
        elbows = [np.array([355.0, 315.0]) for _ in range(frame_count)]
        wrists = [np.array([1200.0, 900.0]) for _ in range(frame_count)]
        hips = [np.array([315.0, 500.0]) for _ in range(frame_count)]
        quality = [0.82] * frame_count

        warning = _anatomy_reliability_warning(
            shoulders, elbows, wrists, hips, quality, shoulder, 92.0, 92.0, 0, frame_count - 1, 30.0
        )

        self.assertIsNotNone(warning)

    def test_barbell_grip_uses_two_hands_from_the_same_lifter_track(self):
        frame_count = 30
        frames = [np.zeros((360, 480), dtype=np.uint8) for _ in range(frame_count)]
        candidates_by_frame = []
        for index in range(frame_count):
            primary_y = 180.0 + index * 2.0
            primary = PoseCandidate(
                index,
                {
                    LEFT_WRIST: np.array([150.0, primary_y]),
                    RIGHT_WRIST: np.array([250.0, primary_y]),
                },
                {LEFT_WRIST: 0.92, RIGHT_WRIST: 0.91},
                np.array([200.0, 250.0]),
                330.0,
                220.0,
            )
            background = PoseCandidate(
                index,
                {
                    LEFT_WRIST: np.array([40.0, 70.0]),
                    RIGHT_WRIST: np.array([120.0, 70.0]),
                },
                {LEFT_WRIST: 0.96, RIGHT_WRIST: 0.96},
                np.array([80.0, 100.0]),
                150.0,
                95.0,
            )
            candidates_by_frame.append([primary, background])

        lifter_track = PersonTrack(track_id=1)
        for frame_candidates in candidates_by_frame:
            lifter_track.append(frame_candidates[0])
        path, confidence, _, confirmed_track = _track_barbell_grip(
            frames, lifter_track, candidates_by_frame
        )

        self.assertIsNotNone(confirmed_track)
        self.assertTrue(_grip_track_is_reliable(path, confidence, 30.0))
        self.assertAlmostEqual(float(path[0][0]), 200.0, delta=2.0)
        self.assertGreater(float(path[-1][1]), float(path[0][1]) + 40.0)

    def test_barbell_grip_rejects_a_track_when_one_hand_disappears(self):
        frame_count = 30
        frames = [np.zeros((360, 480), dtype=np.uint8) for _ in range(frame_count)]
        candidates_by_frame = []
        for index in range(frame_count):
            points = {LEFT_WRIST: np.array([150.0, 180.0 + index])}
            visibility = {LEFT_WRIST: 0.92}
            if not 8 <= index <= 22:
                points[RIGHT_WRIST] = np.array([250.0, 180.0 + index])
                visibility[RIGHT_WRIST] = 0.91
            candidates_by_frame.append([
                PoseCandidate(
                    index,
                    points,
                    visibility,
                    np.array([200.0, 250.0]),
                    330.0,
                    220.0,
                )
            ])

        lifter_track = PersonTrack(track_id=1)
        for frame_candidates in candidates_by_frame:
            lifter_track.append(frame_candidates[0])
        path, confidence, _, _ = _track_barbell_grip(frames, lifter_track, candidates_by_frame)

        self.assertFalse(_grip_track_is_reliable(path, confidence, 30.0))


if __name__ == "__main__":
    unittest.main()