# laikago_toes_zup_inertia.urdf

- Base: Genesis `laikago_toes_zup.urdf` (Z-up, toe contact spheres, joint names match Env).
- Change: fill `<inertial>` (mass/inertia/origin) from PyBullet
  `bullet3/.../laikago_toes_zup.urdf` for matching link names. Genesis copy had
  nearly all-zero body inertias (legacy parser warning).
- Why (#16): Y-up stock/inertia/toes_inertia + RaisimGym pose dies at episode length 11
  under zero-action in LaikagoEnv. Next 1-factor = Z-up toes URDF + identity quat.
- Created: 2026-11-25.
