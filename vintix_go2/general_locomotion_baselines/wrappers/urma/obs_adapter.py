"""Map Genesis loco obs (45) → URMA Go2 flat obs (380) with placeholder descriptions.

Official URMA packs per-joint description (23) + pos/vel/prev_action (3),
per-foot description (10) + contact features (2), then trunk/goal/gravity/
height/context (20). Multi-robot training pads to the max over robot types;
standalone Go2 is **380** (no padding).

Joint **mass** and **joint_range** are filled from
`go2_urdf_description_fields.json` when present. Other description slots
(rel_pos, PD, torque, …) and contact / height remain placeholders until
MuJoCo / Env-side values are available (see NOTES.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .joint_description import (
    FOOT_DESCRIPTION_SIZE,
    FOOT_DESC_LAYOUT,
    GENERAL_OBS_SIZE,
    GO2_ACTION_DIM,
    GO2_NUM_FEET,
    GO2_NUM_JOINTS,
    GO2_OBS_DIM,
    JOINT_DESC_LAYOUT,
    JOINT_DESCRIPTION_SIZE,
    SINGLE_FOOT_OBS,
    SINGLE_JOINT_OBS,
)

_FIELDS_JSON = Path(__file__).resolve().parent / "go2_urdf_description_fields.json"

# Slot offsets inside the 23-dim joint description
_JOINT_SLOT_OFFSET: dict[str, int] = {}
_off = 0
for _name, _n in JOINT_DESC_LAYOUT:
    _JOINT_SLOT_OFFSET[_name] = _off
    _off += _n

_FOOT_SLOT_OFFSET: dict[str, int] = {}
_off = 0
for _name, _n in FOOT_DESC_LAYOUT:
    _FOOT_SLOT_OFFSET[_name] = _off
    _off += _n


def _load_urdf_fields(path: Path = _FIELDS_JSON) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())

GENESIS_SLICES = {
    "ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "commands": slice(6, 9),
    "dof_pos": slice(9, 21),
    "dof_vel": slice(21, 33),
    "actions": slice(33, 45),
}


@dataclass
class GenesisToUrmaGo2Obs:
    batch_size: int = 1
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    # Nominal context placeholders (normalized the same way as URMA comments).
    p_gain: float = 20.0
    d_gain: float = 0.5
    action_scale: float = 0.25
    mass_kg: float = 15.0
    robot_length: float = 0.6
    robot_width: float = 0.3
    robot_height: float = 0.4
    standing_height: float = 0.3
    urdf_fields_path: Path | None = None
    # Filled in __post_init__ from JSON when available.
    _joint_mass: list[float] = field(default_factory=list, repr=False)
    _joint_range: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _foot_mass: list[float] = field(default_factory=list, repr=False)
    _urdf_fields_loaded: bool = field(default=False, repr=False)

    def __post_init__(self):
        self.device = torch.device(self.device)
        self._prev_action = torch.zeros(
            self.batch_size, GO2_ACTION_DIM, device=self.device, dtype=self.dtype
        )
        path = Path(self.urdf_fields_path) if self.urdf_fields_path else _FIELDS_JSON
        data = _load_urdf_fields(path)
        if not data:
            return
        self._urdf_fields_loaded = True
        if "total_mass_kg" in data:
            self.mass_kg = float(data["total_mass_kg"])
        joints = data.get("joints") or []
        self._joint_mass = [float(j.get("mass_kg", 0.0)) for j in joints[:GO2_NUM_JOINTS]]
        self._joint_range = [
            tuple(j.get("joint_range") or (0.0, 0.0)) for j in joints[:GO2_NUM_JOINTS]
        ]
        # Pad if JSON is short
        while len(self._joint_mass) < GO2_NUM_JOINTS:
            self._joint_mass.append(0.0)
            self._joint_range.append((0.0, 0.0))
        feet = data.get("feet") or []
        self._foot_mass = [float(f.get("mass_kg", 0.0)) for f in feet[:GO2_NUM_FEET]]
        while len(self._foot_mass) < GO2_NUM_FEET:
            self._foot_mass.append(0.0)

    def reset(self, batch_idx: slice | None = None):
        if batch_idx is None:
            self._prev_action.zero_()
        else:
            self._prev_action[batch_idx].zero_()

    def _context_block(self, b: int) -> torch.Tensor:
        pd = torch.tensor(
            [
                (self.p_gain / 50.0) - 1.0,
                (self.d_gain / 1.0) - 1.0,
                (self.action_scale / 0.4) - 1.0,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        mass = torch.tensor([(self.mass_kg / 85.0) - 1.0], device=self.device, dtype=self.dtype)
        dims = torch.tensor(
            [
                self.robot_length - 1.0,
                self.robot_width - 1.0,
                self.robot_height - 1.0,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        # Match URMA: (gains / [50,1,0.4])-1 already applied above via /2 then -1 on halves
        # Official: (gains / [100/2, 2/2, 0.8/2]) - 1 = (gains / [50, 1, 0.4]) - 1
        ctx = torch.cat([pd, mass, dims], dim=0)  # 7
        return ctx.unsqueeze(0).expand(b, -1)

    def _joint_desc_placeholder(self, b: int) -> torch.Tensor:
        # Mostly zeros; fill mass + joint_range from URDF JSON when loaded.
        desc = torch.zeros(
            b, GO2_NUM_JOINTS, JOINT_DESCRIPTION_SIZE, device=self.device, dtype=self.dtype
        )
        if not self._urdf_fields_loaded:
            return desc
        mass_i = _JOINT_SLOT_OFFSET["mass"]
        range_i = _JOINT_SLOT_OFFSET["joint_range"]
        for j in range(GO2_NUM_JOINTS):
            lo, hi = self._joint_range[j]
            desc[:, j, range_i] = float(lo)
            desc[:, j, range_i + 1] = float(hi)
            desc[:, j, mass_i] = float(self._joint_mass[j])
        return desc

    def _foot_desc_placeholder(self, b: int) -> torch.Tensor:
        desc = torch.zeros(
            b, GO2_NUM_FEET, FOOT_DESCRIPTION_SIZE, device=self.device, dtype=self.dtype
        )
        if not self._urdf_fields_loaded or not self._foot_mass:
            return desc
        mass_i = _FOOT_SLOT_OFFSET["mass"]
        for f in range(GO2_NUM_FEET):
            desc[:, f, mass_i] = float(self._foot_mass[f])
        return desc

    def step(self, genesis_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if genesis_obs.ndim != 2 or genesis_obs.shape[-1] != 45:
            raise ValueError(f"expected (B, 45) Genesis obs, got {tuple(genesis_obs.shape)}")
        b = genesis_obs.shape[0]
        if b != self.batch_size:
            raise ValueError(f"batch {b} != adapter batch_size {self.batch_size}")

        g = genesis_obs.to(device=self.device, dtype=self.dtype)
        dof_pos = g[:, GENESIS_SLICES["dof_pos"]]
        dof_vel = g[:, GENESIS_SLICES["dof_vel"]]
        actions = g[:, GENESIS_SLICES["actions"]]
        ang = g[:, GENESIS_SLICES["ang_vel"]]
        gravity = g[:, GENESIS_SLICES["projected_gravity"]]
        cmd = g[:, GENESIS_SLICES["commands"]]

        joint_desc = self._joint_desc_placeholder(b)
        # pos is relative to nominal in URMA; Genesis dof_pos is already relative → use as-is
        joint_dyn = torch.stack([dof_pos, dof_vel, self._prev_action], dim=-1)  # (B,12,3)
        joint_block = torch.cat([joint_desc, joint_dyn], dim=-1).reshape(b, -1)  # (B, 312)

        foot_desc = self._foot_desc_placeholder(b)
        foot_extra = torch.zeros(b, GO2_NUM_FEET, 2, device=self.device, dtype=self.dtype)
        foot_block = torch.cat([foot_desc, foot_extra], dim=-1).reshape(b, -1)  # (B, 48)

        # trunk lin vel unknown in Genesis loco obs → zeros; ang vel from sensor
        trunk_lin = torch.zeros(b, 3, device=self.device, dtype=self.dtype)
        height = torch.full(
            (b, 1), self.standing_height, device=self.device, dtype=self.dtype
        )
        ctx = self._context_block(b)
        general = torch.cat(
            [trunk_lin, ang, cmd, gravity, height, ctx],
            dim=-1,
        )  # 3+3+3+3+1+7 = 20

        flat = torch.cat([joint_block, foot_block, general], dim=-1)
        if flat.shape[-1] != GO2_OBS_DIM:
            raise RuntimeError(
                f"built obs dim {flat.shape[-1]} != {GO2_OBS_DIM} "
                f"(joint={SINGLE_JOINT_OBS}, foot={SINGLE_FOOT_OBS}, gen={GENERAL_OBS_SIZE})"
            )

        self._prev_action = actions.detach().clone()
        return {
            "obs_flat": flat,
            "joint_block": joint_block,
            "foot_block": foot_block,
            "general": general,
        }
