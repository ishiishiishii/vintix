#!/usr/bin/env python3
"""CPU-only MorFiC wrapper smoke check (no Genesis / Isaac Gym / GPU).

Usage:
  python -m wrappers.morfic.smoke_cpu_check
  # or from this directory:
  python smoke_cpu_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python smoke_cpu_check.py` from wrappers/morfic/
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]  # general_locomotion_baselines/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wrappers.morfic.policy_loader import (  # noqa: E402
    assert_dims_match_checkpoint,
    load_ac_weight_shapes,
    smoke_forward_shapes,
)


def main() -> int:
    shapes = load_ac_weight_shapes()
    dims = assert_dims_match_checkpoint()
    fwd = smoke_forward_shapes(batch_size=2, steps=5)
    report = {
        "ok": True,
        "dims": dims.__dict__,
        "checkpoint_shapes": {k: list(v) for k, v in shapes.items()},
        "forward_shapes": {k: list(v) for k, v in fwd.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
