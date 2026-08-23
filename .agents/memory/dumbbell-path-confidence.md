---
name: Dumbbell path confidence
description: Product rules for tracking non-circular dumbbells, ending a set, and presenting a path-only score.
---

Hexagonal dumbbells must be tracked from the selected frame-one lifter's hand/handle identity, not with circular plate detection or an unrelated later body track.

**Why:** Circular detection locked onto gym machinery, and post-set lowering plus standing was then miscounted as repetitions. A self-derived single-rep template also produced misleadingly high confidence.

**How to apply:** Associate every wrist observation to the same torso identity, bridge short gaps with optical flow, reject cycles whose lockout/ROM differs from the repeated set pattern, and clear the path after the final retained rep. A dumbbell path score requires at least two reps and compares each rep against the others in normalized 2D; otherwise report the score as unavailable.