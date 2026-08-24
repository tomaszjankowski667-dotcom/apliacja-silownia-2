"""Run the full requested-exercise regression corpus.

Usage:
    python -m vision.test.run_video_corpus
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from vision.exercise_compatibility import ExerciseMismatchError
from vision.vision_exercise_analyzer import analyze_video_with_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("video_corpus_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all or a slice of the requested-exercise video corpus."
    )
    parser.add_argument("--start", type=int, default=0, help="Inclusive manifest index.")
    parser.add_argument("--end", type=int, help="Exclusive manifest index.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    all_recordings = manifest["recordings"]
    end = len(all_recordings) if args.end is None else args.end
    recordings = all_recordings[args.start:end]
    output_dir = PROJECT_ROOT / ".local" / "corpus-results"
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    rows: list[dict] = []
    for item in recordings:
        source = PROJECT_ROOT / "attached_assets" / item["file"]
        output = output_dir / item["file"]
        rejection_code = None
        rejection_evidence = None
        compatibility_summary = None
        error_type = None
        try:
            with redirect_stdout(StringIO()):
                summary = analyze_video_with_model(
                    source,
                    output,
                    manifest["exercise"],
                    allow_uncertain_equipment=item.get(
                        "allow_uncertain_equipment",
                        False,
                    ),
                )
            accepted = True
            compatibility_summary = summary.get("exercise_compatibility")
            detail = f"{summary['completed_reps']} reps"
            expected_reps = item.get("completed_reps")
            expected_compatibility_codes = set(
                item.get("compatibility_codes", [])
            )
            compatibility_code_ok = (
                not expected_compatibility_codes
                or (
                    compatibility_summary is not None
                    and compatibility_summary.get("code")
                    in expected_compatibility_codes
                )
            )
            bar_path_only_ok = (
                not item.get("require_bar_path_only", False)
                or (
                    summary.get("analysis_mode") == "bar_path_only"
                    and summary.get("limb_motion_score") is None
                    and summary.get("final_score") is None
                )
            )
            passed = (
                bool(item["compatible"])
                and compatibility_code_ok
                and bar_path_only_ok
                and (
                    expected_reps is None
                    or summary["completed_reps"] == expected_reps
                )
            )
        except ExerciseMismatchError as error:
            accepted = False
            rejection_code = error.decision.code
            rejection_evidence = error.decision.to_dict()["evidence"]
            detail = str(error)
            output.unlink(missing_ok=True)
            expected_codes = set(item.get("rejection_codes", []))
            passed = (
                not bool(item["compatible"])
                and (not expected_codes or rejection_code in expected_codes)
            )
        except Exception as error:
            accepted = False
            error_type = type(error).__name__
            detail = str(error)
            output.unlink(missing_ok=True)
            passed = False
        rows.append(
            {
                "file": item["file"],
                "expected_compatible": item["compatible"],
                "allow_uncertain_equipment": item.get(
                    "allow_uncertain_equipment",
                    False,
                ),
                "require_bar_path_only": item.get(
                    "require_bar_path_only",
                    False,
                ),
                "accepted": accepted,
                "passed": passed,
                "detail": detail,
                "completed_reps": (
                    summary["completed_reps"] if accepted else None
                ),
                "rejection_code": rejection_code,
                "rejection_evidence": rejection_evidence,
                "compatibility": compatibility_summary,
                "error_type": error_type,
            }
        )
        if not passed:
            failures.append(item["file"])
        print(
            f"{'PASS' if passed else 'FAIL'} {item['file']}: "
            f"{'accepted' if accepted else 'rejected'} — {detail}"
        )
    report_path = output_dir / f"report-{args.start}-{end}.json"
    report_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())