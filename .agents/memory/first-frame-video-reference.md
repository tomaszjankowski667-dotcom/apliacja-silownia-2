---
name: First-frame video reference
description: Product rule for anchoring exercise-video trajectories and handling an occluded lifter.
---

The first automatically validated analysis frame is the immutable equipment origin. Later smoothing, repetition boundaries, or clip-wide statistics must not recenter that reference.

**Why:** Recomputing the origin during a set made the target and overlay drift, while asking users to hand-trim an unusable video lead-in created avoidable friction.

**How to apply:** Before analysis, scan the opening seconds for the earliest validated barbell plate; trim only that untrusted lead-in, then lock the origin. If the plate is not a reliable circle, use a high-coverage midpoint of both directly observed hands from one unambiguous lifter track. For hex dumbbells, anchor the selected lifter's hand/handle. If anatomy is ambiguous, keep the equipment reference but downgrade to bar-path-only and omit inferred anatomy.