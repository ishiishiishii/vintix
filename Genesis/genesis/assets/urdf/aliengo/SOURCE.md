# Aliengo URDF source

- Upstream: https://github.com/unitreerobotics/unitree_ros
- Path: `robots/aliengo_description/`
- License: BSD-3-Clause (see `LICENSE`)
- Imported for Issue #16 (2026-11-18)
- URDF total mass (sum of link masses): ≈24.94 kg

## Visual mesh note (2026-11-23)

`meshes/calf.dae` declares COLLADA `<unit meter="0.025400"/>` (inch).
Genesis/Assimp path used here does not apply that unit, so URDF visual scale is set to `0.0254` for all four calf meshes. Collision boxes unchanged.
