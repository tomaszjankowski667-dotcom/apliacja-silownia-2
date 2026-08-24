---
name: Vision confidence policy
description: Reliability rule for exercise-video analysis under occlusion, multiple people, and camera ambiguity.
---

Exercise-video analysis must reject ambiguous subject selection, camera-side selection, equipment identity, anatomy, or repetition boundaries rather than score inferred data as fact.

**Why:** Gym footage can include spotters, plates that hide joints, and background circular objects. A plausible-looking score from the wrong person or object is worse than a clear request for a better clip.

**How to apply:** Keep confidence gates independent across pose, anatomy, and equipment tracking. Compatibility evidence must come from one continuous person track; never combine favorable frames from different people. Pose-derived hand motion may assist tracking but cannot prove equipment identity. A single round load confirms a barbell only with a visually long bar lever or repeated shared-shaft evidence through both hands; an occluded axial view without either is unresolved and must fail closed. Full anatomy requires a complete frame-one shoulder–elbow–wrist–hip observation plus continuous, calibrated arm-chain and body-scale evidence during the set; this rule is exercise-agnostic. Do not render reconstructed joints as observed anatomy. When anatomy is ambiguous but equipment is unambiguous, label the result as bar-path-only rather than presenting a full anatomical technique score.