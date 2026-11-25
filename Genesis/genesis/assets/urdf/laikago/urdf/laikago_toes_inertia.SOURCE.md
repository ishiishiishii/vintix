# laikago_toes_inertia.urdf

- Base: Genesis `laikago_toes.urdf` (Y-up, same joint names as stock, **adds toe contact links**).
- Change (1 factor for #16 after inertia-only failed): copy non-zero `<inertia>` diagonals
  from `laikago_inertia.urdf` / PyBullet toes_zup onto the body links. Toe link inertias
  already non-zero in toes.urdf.
- Why: inertia-only patch on stock (no toes) still collapsed (mean length ≈5). Standing
  render historically used toes_zup; missing foot spheres may hurt contact.
- Created: 2026-11-25 for expert retrain (#16).
