"""Registry describing exercises supported by the video analyzer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExerciseSpec:
    key: str
    template_module: str
    tracking_mode: str
    compatibility_policy: str
    video_supported: bool = True


_MID_CHEST_MODULE = "analyze.analyze_mid_chest"

EXERCISE_REGISTRY: dict[str, ExerciseSpec] = {
    "Flat_Barbell_Press": ExerciseSpec(
        key="Flat_Barbell_Press",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="barbell",
        compatibility_policy="flat_barbell_press",
    ),
    "Flat_Dumbbell_Press": ExerciseSpec(
        key="Flat_Dumbbell_Press",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="dumbbell",
        compatibility_policy="flat_dumbbell_press",
        video_supported=True,
    ),
    "Machine_Chest_Press": ExerciseSpec(
        key="Machine_Chest_Press",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="unsupported",
        compatibility_policy="",
        video_supported=False,
    ),
    "Flat_Smith_Press": ExerciseSpec(
        key="Flat_Smith_Press",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="unsupported",
        compatibility_policy="",
        video_supported=False,
    ),
    "Mid_Cable_Crossover": ExerciseSpec(
        key="Mid_Cable_Crossover",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="unsupported",
        compatibility_policy="",
        video_supported=False,
    ),
    "Butterfly_Pec_Deck": ExerciseSpec(
        key="Butterfly_Pec_Deck",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="unsupported",
        compatibility_policy="",
        video_supported=False,
    ),
    "Flat_Dumbbell_Flyes": ExerciseSpec(
        key="Flat_Dumbbell_Flyes",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="dumbbell",
        compatibility_policy="",
        video_supported=False,
    ),
    "Hex_Squeeze_Press": ExerciseSpec(
        key="Hex_Squeeze_Press",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="dumbbell",
        compatibility_policy="",
        video_supported=False,
    ),
    "Push_Ups_Standard": ExerciseSpec(
        key="Push_Ups_Standard",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="unsupported",
        compatibility_policy="",
        video_supported=False,
    ),
    "Chest_Dips": ExerciseSpec(
        key="Chest_Dips",
        template_module=_MID_CHEST_MODULE,
        tracking_mode="unsupported",
        compatibility_policy="",
        video_supported=False,
    ),
}


def get_exercise_spec(exercise_key: str) -> ExerciseSpec:
    try:
        spec = EXERCISE_REGISTRY[exercise_key]
    except KeyError as error:
        available = ", ".join(sorted(EXERCISE_REGISTRY))
        raise ValueError(
            f"Unknown exercise '{exercise_key}'. Available template keys: {available}"
        ) from error
    if not spec.video_supported:
        supported = ", ".join(
            key
            for key, candidate in EXERCISE_REGISTRY.items()
            if candidate.video_supported
        )
        raise ValueError(
            f"Exercise '{exercise_key}' has a biomechanical template but is not yet "
            "supported by the fail-closed video analyzer. "
            f"Currently supported: {supported}."
        )
    return spec