# laikago_inertia.urdf

- Base: Genesis `laikago.urdf` (Y-up, joint names `FR_hip_motor_2_chassis_joint` …).
- Change (1 factor for #16): `<inertia>` diagonal values copied from PyBullet
  `bullet3/.../pybullet_data/laikago/laikago_toes_zup.urdf` for matching link names
  (chassis / hip_motor / upper_leg / lower_leg). Masses and joint graph unchanged.
- Why: stock Genesis/PyBullet `laikago.urdf` has all-zero inertia tensors; PPO
  collapses (mean episode length ≈5–8). toes_zup has non-zero inertia but adds toe
  links and is Z-up — so we only transplant inertia into the existing Y-up URDF.
- Created: 2026-11-25 for expert retrain (#16).
