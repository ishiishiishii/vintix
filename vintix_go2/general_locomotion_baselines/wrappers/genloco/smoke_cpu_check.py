#!/usr/bin/env python3
"""CPU-only GenLoco wrapper smoke check (no Genesis / PyBullet / GPU).

Usage:
  python wrappers/genloco/smoke_cpu_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wrappers.genloco.policy_loader import (  # noqa: E402
    assert_all_policy_zips_match,
    assert_dims_match_checkpoint,
    load_weight_shapes,
    smoke_forward_shapes,
)


def main() -> int:
    shapes = load_weight_shapes()
    dims = assert_dims_match_checkpoint()
    checked = assert_all_policy_zips_match()
    fwd = smoke_forward_shapes(batch_size=2, steps=5)
    report = {
        "ok": True,
        "dims": dims.__dict__,
        "checkpoint_shapes": {k: list(v) for k, v in shapes.items()},
        "policy_zips_checked": checked,
        "forward_shapes": {k: list(v) for k, v in fwd.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
