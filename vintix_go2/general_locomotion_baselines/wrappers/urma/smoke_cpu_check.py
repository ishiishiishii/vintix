#!/usr/bin/env python3
"""CPU-only URMA Go2 obs-geometry smoke (no MuJoCo / RL-X / GPU).

Usage:
  python wrappers/urma/smoke_cpu_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wrappers.urma.policy_loader import (  # noqa: E402
    assert_go2_dims,
    inspect_checkpoint,
    smoke_obs_shapes,
)


def main() -> int:
    dims = assert_go2_dims()
    info = inspect_checkpoint()
    fwd = smoke_obs_shapes(batch_size=2, steps=5)
    report = {
        "ok": True,
        "dims": dims.__dict__,
        "checkpoint": {
            "path": info.path,
            "size_bytes": info.size_bytes,
            "magic_ok": info.magic_ok,
            "format_note": info.format_note,
        },
        "forward_shapes": {k: list(v) for k, v in fwd.items()},
        "actor_forward": "deferred (needs RL-X)",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
