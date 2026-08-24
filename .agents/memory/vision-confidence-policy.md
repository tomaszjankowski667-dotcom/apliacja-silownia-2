---
name: Vision confidence policy
description: Reliability rule for exercise-video analysis under occlusion, multiple people, and camera ambiguity.
---

Exercise-video analysis must reject hard exercise/equipment mismatches and must never score ambiguous anatomy as fact. Technique defects belong in diagnostics, not compatibility rejection. A user may explicitly continue predefined equipment ambiguities, but only with equipment-path scoring.

**Why:** Gym footage can include spotters, plates that hide joints, and background circular objects, while genuinely bad form is exactly what the product must diagnose. Treating poor technique as another exercise defeats the product; inventing anatomy or accepting machine guides as a free bar is equally misleading.

**How to apply:** Keep confidence gates independent across pose, anatomy, equipment identity, and technique. Compatibility evidence must come from one continuous person track; never combine favorable frames from different people. Pose-derived hand motion may assist tracking but cannot prove equipment identity. Distinguish free-bar endpoints from Smith guides and machine arms using projection beyond the two-hand grip plus repeated plate–grip collinearity. Never let an uncertainty override bypass a confirmed guided/machine geometry or another hard mismatch. In explicit uncertainty mode, show a prominent warning and force limb and overall scores unavailable. Full anatomy requires a complete frame-one shoulder–elbow–wrist–hip observation plus continuous, calibrated arm-chain and body-scale evidence during the set; do not render reconstructed joints as observed anatomy.