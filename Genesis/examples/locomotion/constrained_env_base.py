"""
Base class for constrained environments that separate rewards and costs.
This implements the approach from "Not Only Rewards But Also Constraints" paper.

The key idea is to:
1. Separate rewards (maximize) from costs (constraints to satisfy)
2. Define two types of costs:
   - Probabilistic constraints: binary indicators (0 if satisfied, 1 if violated)
   - Average constraints: continuous values to minimize on average
"""

import torch
import genesis as gs


class ConstrainedEnvMixin:
    """
    Mixin class that adds constraint (cost) computation to existing environments.
    """
    
    def __init__(self, *args, cost_cfg=None, **kwargs):
        """
        Initialize cost functions.
        
        Args:
            cost_cfg: Configuration dictionary with:
                - cost_functions: dict of cost function names to use
                - cost_types: dict mapping cost names to 'probabilistic' or 'average'
                - cost_thresholds: dict mapping cost names to constraint thresholds (d_k in paper)
        """
        super().__init__(*args, **kwargs)
        
        # Cost configuration
        self.cost_cfg = cost_cfg or {}
        self.cost_functions = {}
        self.cost_types = self.cost_cfg.get("cost_types", {})  # 'probabilistic' or 'average'
        self.cost_thresholds = self.cost_cfg.get("cost_thresholds", {})  # d_k in paper
        
        # Initialize cost functions
        cost_func_names = self.cost_cfg.get("cost_functions", {})
        self.cost_episode_sums = {}
        
        for name in cost_func_names.keys():
            if hasattr(self, f"_cost_{name}"):
                self.cost_functions[name] = getattr(self, f"_cost_" + name)
                self.cost_episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
            else:
                print(f"Warning: Cost function '_cost_{name}' not found, skipping.")
        
        # Cost buffer (shape: [num_envs, num_costs])
        num_costs = len(self.cost_functions)
        self.cost_buf = torch.zeros((self.num_envs, num_costs), device=gs.device, dtype=gs.tc_float)
    
    def compute_costs(self):
        """
        Compute all costs and store in cost_buf.
        
        Returns:
            cost_buf: Tensor of shape [num_envs, num_costs]
        """
        self.cost_buf[:] = 0.0
        
        for idx, (name, cost_func) in enumerate(self.cost_functions.items()):
            cost = cost_func()
            
            # For probabilistic constraints, ensure binary output
            if self.cost_types.get(name, "average") == "probabilistic":
                cost = (cost > 0.0).float()  # Binary indicator
            
            # Multiply by dt for time normalization (if needed)
            cost = cost * self.dt
            
            self.cost_buf[:, idx] = cost
            self.cost_episode_sums[name] += cost
        
        return self.cost_buf
    
    def step(self, actions):
        """
        Override step to compute and return costs.
        
        Returns:
            obs, reward, reset_buf, extras (with extras['cost'] added)
        """
        # Call parent step
        obs, reward, reset_buf, extras = super().step(actions)
        
        # Compute costs
        costs = self.compute_costs()
        
        # Add costs to extras for logging
        extras['cost'] = costs  # Shape: [num_envs, num_costs]
        extras['cost_names'] = list(self.cost_functions.keys())
        
        # Log cost episode sums in reset_idx
        return obs, reward, reset_buf, extras
    
    def reset_idx(self, envs_idx):
        """
        Override to reset cost episode sums.
        """
        super().reset_idx(envs_idx)
        
        # Reset cost episode sums and log them
        for name in self.cost_functions.keys():
            if "episode" not in self.extras:
                self.extras["episode"] = {}
            
            self.extras["episode"]["cost_" + name] = (
                torch.mean(self.cost_episode_sums[name][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.cost_episode_sums[name][envs_idx] = 0.0


# Note: Cost functions should be implemented in the concrete environment classes
# that inherit from ConstrainedEnvMixin. See train_constrained.py for examples.

