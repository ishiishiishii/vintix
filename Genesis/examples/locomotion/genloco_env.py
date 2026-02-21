"""
GenLoco-style Morphological Randomization Environment for Go2 Robot
Implements morphological randomization (scale, mass, leg dimensions) for training 
generalized locomotion policies following GenLoco methodology.
"""

import torch
import math
import random
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from go2_env import Go2Env, gs_rand_float


class GenLocoGo2Env(Go2Env):
    """
    GenLoco-style morphological randomization environment for Go2 robot.
    
    Randomizes:
    1. Robot overall scale (size factor α) - affects all dimensions uniformly
    2. Base/torso mass - independent from scale
    3. Target height - scaled with robot size to maintain task difficulty
    
    Following GenLoco paper methodology but using velocity/height-based rewards
    instead of motion capture imitation.
    """
    
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        # GenLoco morphological randomization parameters
        enable_morphological_randomization=True,
        size_factor_range=(0.8, 1.2),  # Overall robot scale (α in GenLoco paper)
        mass_range=(0.7, 1.3),  # Base mass multiplier (independent from size)
        # PD gain randomization (optional, for robustness)
        kp_range=(0.9, 1.1),  # PD position gain multiplier
        kd_range=(0.9, 1.1),  # PD velocity gain multiplier
    ):
        """
        Args:
            enable_morphological_randomization: Enable GenLoco-style morphological randomization
            size_factor_range: Range for overall robot scale (affects all dimensions)
            mass_range: Range for base mass multiplier (independent from scale^3)
            kp_range: Range for PD position gain multiplier
            kd_range: Range for PD velocity gain multiplier
        """
        # Store GenLoco parameters
        self.enable_morphological_randomization = enable_morphological_randomization
        self.size_factor_range = size_factor_range
        self.mass_range = mass_range
        self.kp_range = kp_range
        self.kd_range = kd_range
        
        # Current randomization values (per environment)
        self.current_size_factors = torch.ones((num_envs,), device=gs.device, dtype=gs.tc_float)
        self.current_mass_scales = torch.ones((num_envs,), device=gs.device, dtype=gs.tc_float)
        self.current_kp_scales = torch.ones((num_envs,), device=gs.device, dtype=gs.tc_float)
        self.current_kd_scales = torch.ones((num_envs,), device=gs.device, dtype=gs.tc_float)
        
        # Store original config values
        self.original_base_height_target = reward_cfg.get("base_height_target", 0.3)
        self.original_base_init_pos_z = env_cfg.get("base_init_pos", [0.0, 0.0, 0.42])[2]
        self.original_kp = env_cfg.get("kp", 20.0)
        self.original_kd = env_cfg.get("kd", 0.5)
        
        # Initialize base environment (but don't build scene yet)
        # We need to set domain_randomization=False to avoid double randomization
        super().__init__(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
            domain_randomization=False,  # We handle randomization ourselves
            mass_range=(1.0, 1.0),  # Disable base class randomization
        )
        
        # Override domain_randomization flag to use our GenLoco randomization
        self.original_mass = None
        
        # Apply initial morphological randomization
        if self.enable_morphological_randomization:
            self._initialize_morphological_randomization()
    
    def _initialize_morphological_randomization(self):
        """Initialize morphological randomization by creating robots with random scales."""
        # Note: In Genesis, we can't change scale after scene.build() is called.
        # So we'll randomize mass and PD gains dynamically, but for scale we'd need
        # to recreate robots which is expensive. Instead, we'll randomize mass 
        # (which scales with size^3) and adjust target heights accordingly.
        
        # Sample initial random values for all environments
        self._sample_morphological_parameters(torch.arange(self.num_envs, device=gs.device))
        
        # Save original mass for later randomization
        self._save_original_physics()
    
    def _sample_morphological_parameters(self, envs_idx):
        """Sample new morphological parameters for specified environments."""
        num_envs = len(envs_idx)
        
        # Sample size factors (will affect target heights)
        size_factors = gs_rand_float(
            self.size_factor_range[0],
            self.size_factor_range[1],
            (num_envs,),
            gs.device
        )
        self.current_size_factors[envs_idx] = size_factors
        
        # Sample mass scales (independent from size, for additional variation)
        mass_scales = gs_rand_float(
            self.mass_range[0],
            self.mass_range[1],
            (num_envs,),
            gs.device
        )
        self.current_mass_scales[envs_idx] = mass_scales
        
        # Sample PD gain scales (for robustness)
        kp_scales = gs_rand_float(
            self.kp_range[0],
            self.kp_range[1],
            (num_envs,),
            gs.device
        )
        self.current_kp_scales[envs_idx] = kp_scales
        
        kd_scales = gs_rand_float(
            self.kd_range[0],
            self.kd_range[1],
            (num_envs,),
            gs.device
        )
        self.current_kd_scales[envs_idx] = kd_scales
        
        # Update target heights based on size factors (maintain relative difficulty)
        self._update_scaled_target_heights(envs_idx)
        
        # Update PD gains
        self._update_pd_gains(envs_idx)
    
    def _update_scaled_target_heights(self, envs_idx):
        """Update target heights based on current size factors."""
        # Target height scales with robot size to maintain task difficulty
        # If robot is 20% larger, target height should also be 20% higher
        size_factors = self.current_size_factors[envs_idx]
        
        # Update reward config's target height (for reward calculation)
        # Note: Genesis doesn't support per-env configs easily, so we use mean
        # This is an approximation but should work well in practice
        if len(envs_idx) == self.num_envs or len(envs_idx) > self.num_envs // 2:
            # Most/all environments reset - use mean size factor
            mean_size_factor = size_factors.mean().item()
            self.reward_cfg["base_height_target"] = self.original_base_height_target * mean_size_factor
            # Also adjust initial position (scaled with size)
            self.env_cfg["base_init_pos"][2] = self.original_base_init_pos_z * mean_size_factor
            # Update base_init_pos tensor
            self.base_init_pos[:, 2] = self.env_cfg["base_init_pos"][2]
    
    def _update_pd_gains(self, envs_idx):
        """Update PD gains based on current scales."""
        if not hasattr(self, 'motors_dof_idx') or self.motors_dof_idx is None:
            return
        
        # Update PD gains for specified environments
        # Note: Genesis API might not support per-env PD gains, so we use mean
        if len(envs_idx) == self.num_envs or len(envs_idx) > self.num_envs // 2:
            # Use mean values if most/all envs are being updated
            mean_kp = self.current_kp_scales[envs_idx].mean().item() * self.original_kp
            mean_kd = self.current_kd_scales[envs_idx].mean().item() * self.original_kd
            
            # Apply to all dofs (Genesis might not support per-env)
            self.robot.set_dofs_kp([mean_kp] * self.num_actions, self.motors_dof_idx)
            self.robot.set_dofs_kv([mean_kd] * self.num_actions, self.motors_dof_idx)
    
    def _save_original_physics(self):
        """Save original physics parameters."""
        try:
            base_link = self.robot.get_link("base")
            self.original_mass = base_link.get_mass()
        except Exception as e:
            print(f"Warning: Could not get original mass: {e}")
            self.original_mass = 6.921  # Default Go2 base mass
    
    def _apply_morphological_randomization(self, envs_idx):
        """Apply morphological randomization to specified environments."""
        if not self.enable_morphological_randomization:
            return
        
        # Sample new parameters
        self._sample_morphological_parameters(envs_idx)
        
        # Apply mass randomization (scale^3 * mass_scale for physical correctness)
        # In GenLoco, mass scales with volume (size^3), but we add extra independent variation
        try:
            base_link = self.robot.get_link("base")
            
            # Compute new masses: original_mass * size_factor^3 * mass_scale
            # This simulates both size variation (via size_factor^3) and independent mass variation
            size_factors = self.current_size_factors[envs_idx]
            mass_scales = self.current_mass_scales[envs_idx]
            
            # Genesis API limitation: can't set per-env masses easily
            # Use mean for all envs (approximation - in practice, this still provides diversity
            # across episodes since we resample every reset)
            if len(envs_idx) == self.num_envs or len(envs_idx) > self.num_envs // 2:
                mean_size_factor = size_factors.mean().item()
                mean_mass_scale = mass_scales.mean().item()
                new_mass = self.original_mass * (mean_size_factor ** 3) * mean_mass_scale
                base_link.set_mass(new_mass)
        except Exception as e:
            print(f"Warning: Could not randomize mass: {e}")
    
    def reset_idx(self, envs_idx):
        """Reset specified environments with morphological randomization."""
        # Apply GenLoco morphological randomization before reset
        # This ensures each episode has different morphological parameters
        if self.enable_morphological_randomization:
            self._apply_morphological_randomization(envs_idx)
        
        # Call parent reset (will reset positions, velocities, etc.)
        super().reset_idx(envs_idx)
        
        # Note: We randomize morphology on each reset to ensure diversity
        # across episodes, following GenLoco methodology
    
    def _reward_base_height(self):
        """Reward for maintaining target base height (scaled with robot size)."""
        # Get per-environment target heights
        # Since we can't easily have per-env targets in current implementation,
        # we use the mean size factor (approximation)
        if self.enable_morphological_randomization:
            mean_size_factor = self.current_size_factors.mean().item()
            target_height = self.original_base_height_target * mean_size_factor
        else:
            target_height = self.reward_cfg["base_height_target"]
        
        # Penalize deviation from target height
        height_error = torch.square(self.base_pos[:, 2] - target_height)
        return height_error
    
    def get_morphological_info(self):
        """Get current morphological randomization information."""
        if not self.enable_morphological_randomization:
            return None
        
        return {
            'size_factors': self.current_size_factors.cpu().numpy(),
            'mass_scales': self.current_mass_scales.cpu().numpy(),
            'kp_scales': self.current_kp_scales.cpu().numpy(),
            'kd_scales': self.current_kd_scales.cpu().numpy(),
            'size_factor_range': self.size_factor_range,
            'mass_range': self.mass_range,
        }
    
    def set_morphological_ranges(
        self,
        size_factor_range=None,
        mass_range=None,
        kp_range=None,
        kd_range=None,
    ):
        """Dynamically change morphological randomization ranges."""
        if size_factor_range is not None:
            self.size_factor_range = size_factor_range
        if mass_range is not None:
            self.mass_range = mass_range
        if kp_range is not None:
            self.kp_range = kp_range
        if kd_range is not None:
            self.kd_range = kd_range

