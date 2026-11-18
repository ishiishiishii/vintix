"""MorFiC morphology descriptor (copied keys/ranges; no third_party import).

Upstream: prakharmishra27/morfic-rl `go2_gym/envs/robots/morphology.py` (MIT).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

MORPHOLOGY_KEYS = (
    "thigh_mass",
    "calf_mass",
    "hip_mass",
    "foot_mass",
    "base_mass",
    "hip_limit_min",
    "hip_limit_max",
    "thigh_limit_min",
    "thigh_limit_max",
    "calf_limit_min",
    "calf_limit_max",
)

# Training bounds from MorFiC (used to normalize descriptors).
MORPHOLOGY_RANGES = {
    "thigh_mass": (0.634, 1.152),
    "calf_mass": (0.064, 0.166),
    "hip_mass": (0.510299, 0.696),
    "foot_mass": (0.0, 0.06),
    "base_mass": (3.3, 6.921),
    "hip_limit_min": (-1.10, -0.80),
    "hip_limit_max": (0.80, 1.10),
    "thigh_limit_min": (-1.65, -1.05),
    "thigh_limit_max": (3.49, 4.05),
    "calf_limit_min": (-2.72, -2.60),
    "calf_limit_max": (-0.95, -0.78),
}


def morphology_bounds(device=None, dtype=torch.float32):
    lo = torch.tensor(
        [MORPHOLOGY_RANGES[k][0] for k in MORPHOLOGY_KEYS],
        dtype=dtype,
        device=device,
    )
    hi = torch.tensor(
        [MORPHOLOGY_RANGES[k][1] for k in MORPHOLOGY_KEYS],
        dtype=dtype,
        device=device,
    )
    return lo, hi


def go2_nominal_morphology(device=None, dtype=torch.float32) -> torch.Tensor:
    """Mid-range descriptor inside MorFiC training bounds (placeholder for Go2)."""
    lo, hi = morphology_bounds(device=device, dtype=dtype)
    return 0.5 * (lo + hi)


def normalize_morphology(
    raw: torch.Tensor | Sequence[float],
    morph_min: torch.Tensor | None = None,
    morph_max: torch.Tensor | None = None,
) -> torch.Tensor:
    if morph_min is None or morph_max is None:
        morph_min, morph_max = morphology_bounds(
            device=getattr(raw, "device", None),
            dtype=torch.float32,
        )
    raw_t = torch.as_tensor(raw, dtype=morph_min.dtype, device=morph_min.device)
    if raw_t.shape[-1] != len(MORPHOLOGY_KEYS):
        raise ValueError(f"expected last dim {len(MORPHOLOGY_KEYS)}, got {raw_t.shape}")
    return (raw_t - morph_min) / (morph_max - morph_min)


def dict_to_vector(values: dict, device=None, dtype=torch.float32) -> torch.Tensor:
    missing = [k for k in MORPHOLOGY_KEYS if k not in values]
    if missing:
        raise KeyError(f"missing morphology keys: {missing}")
    return torch.tensor(
        [float(values[k]) for k in MORPHOLOGY_KEYS],
        dtype=dtype,
        device=device,
    )
