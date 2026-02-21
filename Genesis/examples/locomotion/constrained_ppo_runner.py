"""
Constrained PPO Runner for rsl-rl-lib
Extends OnPolicyRunner to support constraint-based RL with cost critics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from constrained_ppo import MultiHeadCostCritic

import genesis as gs


class ConstrainedActorCritic(ActorCritic):
    """
    ActorCritic with additional Cost Critic.
    Extends rsl-rl-lib's ActorCritic to include cost value estimation.
    """
    
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, num_costs, **kwargs):
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        
        # Cost Critic network
        cost_critic_hidden_dims = kwargs.get("cost_critic_hidden_dims", kwargs.get("critic_hidden_dims", [512, 256, 128]))
        activation = kwargs.get("activation", "elu")
        
        self.cost_critic = MultiHeadCostCritic(
            num_obs=num_critic_obs,
            num_costs=num_costs,
            hidden_dims=cost_critic_hidden_dims,
            activation=activation,
        )
        
        # Cost critic optimizer
        cost_lr = kwargs.get("cost_learning_rate", kwargs.get("learning_rate", 0.001))
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=cost_lr)
    
    def evaluate_cost(self, obs):
        """Evaluate cost values for given observations."""
        return self.cost_critic(obs)


class ConstrainedPPOAlgorithm:
    """
    Constrained PPO Algorithm that extends standard PPO with constraint handling.
    This class works alongside rsl-rl-lib's PPO but adds constraint penalty logic.
    """
    
    def __init__(
        self,
        num_costs: int,
        cost_thresholds: Dict[str, float],
        cost_names: List[str],
        cost_types: Dict[str, str],
        penalty_coef: float = 1.0,
        adaptive_threshold: bool = True,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cuda",
    ):
        self.num_costs = num_costs
        self.cost_thresholds = cost_thresholds
        self.cost_names = cost_names
        self.cost_types = cost_types
        self.penalty_coef = penalty_coef
        self.adaptive_threshold = adaptive_threshold
        self.gamma = gamma
        self.lam = lam
        self.device = device
    
    def compute_cost_gae(
        self,
        costs: torch.Tensor,  # [num_steps, num_envs, num_costs]
        cost_values: torch.Tensor,  # [num_steps, num_envs, num_costs]
        dones: torch.Tensor,  # [num_steps, num_envs]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation) for costs.
        
        Returns:
            cost_advantages: [num_steps, num_envs, num_costs]
            cost_returns: [num_steps, num_envs, num_costs]
        """
        num_steps, num_envs, num_costs = costs.shape
        
        cost_advantages = torch.zeros_like(costs)
        cost_returns = torch.zeros_like(costs)
        
        for cost_idx in range(num_costs):
            last_gae_cost = 0
            
            for step in reversed(range(num_steps)):
                if step == num_steps - 1:
                    next_cost_value = cost_values[step, :, cost_idx]
                    next_done = dones[step]
                else:
                    next_cost_value = cost_values[step + 1, :, cost_idx]
                    next_done = dones[step + 1]
                
                delta_cost = (
                    costs[step, :, cost_idx]
                    + self.gamma * next_cost_value * (1 - next_done)
                    - cost_values[step, :, cost_idx]
                )
                
                cost_advantages[step, :, cost_idx] = last_gae_cost = (
                    delta_cost + self.gamma * self.lam * (1 - next_done) * last_gae_cost
                )
            
            cost_returns[:, :, cost_idx] = cost_advantages[:, :, cost_idx] + cost_values[:, :, cost_idx]
        
        return cost_advantages, cost_returns
    
    def compute_constraint_penalty(self, cost_returns_mean: torch.Tensor) -> float:
        """
        Compute log barrier penalty term.
        
        Args:
            cost_returns_mean: Mean cost returns for current batch [num_costs]
        
        Returns:
            penalty: Scalar penalty value
        """
        penalty = 0.0
        
        for idx, cost_name in enumerate(self.cost_names):
            d_k = self.cost_thresholds.get(cost_name, 1.0)
            J_c_k = cost_returns_mean[idx].item()
            
            # Adaptive thresholding
            if self.adaptive_threshold and J_c_k > d_k:
                d_k_adaptive = max(d_k, J_c_k * 1.01)
            else:
                d_k_adaptive = d_k
            
            # Log barrier: -log(d_k - J_c_k)
            epsilon = 1e-6
            barrier_arg = d_k_adaptive - J_c_k + epsilon
            
            if barrier_arg > 0:
                penalty_val = -self.penalty_coef * np.log(barrier_arg)
            else:
                penalty_val = self.penalty_coef * 100.0  # Large penalty for severe violation
            
            penalty += penalty_val
        
        return penalty
    
    def update_cost_critic(
        self,
        critic_obs: torch.Tensor,
        cost_returns: torch.Tensor,  # [batch, num_costs]
        cost_values: torch.Tensor,  # [batch, num_costs]
        cost_critic,
        cost_critic_optimizer,
        use_clipped_value_loss: bool = True,
        clip_param: float = 0.2,
    ) -> float:
        """
        Update cost critic network.
        
        Returns:
            cost_value_loss: Average cost value loss
        """
        # Predict new cost values
        new_cost_values = cost_critic(critic_obs)  # [batch, num_costs]
        
        # Compute cost value loss for each cost
        cost_value_loss_total = 0.0
        
        for cost_idx in range(self.num_costs):
            cost_target = cost_returns[:, cost_idx]
            cost_pred = new_cost_values[:, cost_idx]
            old_cost_pred = cost_values[:, cost_idx]
            
            if use_clipped_value_loss:
                value_pred_clipped = old_cost_pred + torch.clamp(
                    cost_pred - old_cost_pred, -clip_param, clip_param
                )
                value_losses_unclipped = (cost_pred - cost_target) ** 2
                value_losses_clipped = (value_pred_clipped - cost_target) ** 2
                cost_value_loss = 0.5 * torch.max(value_losses_unclipped, value_losses_clipped).mean()
            else:
                cost_value_loss = 0.5 * ((cost_pred - cost_target) ** 2).mean()
            
            cost_value_loss_total += cost_value_loss
        
        # Average over costs
        cost_value_loss_avg = cost_value_loss_total / self.num_costs
        
        # Update
        cost_critic_optimizer.zero_grad()
        cost_value_loss_avg.backward()
        torch.nn.utils.clip_grad_norm_(cost_critic.parameters(), 1.0)
        cost_critic_optimizer.step()
        
        return cost_value_loss_avg.item()


class ConstrainedOnPolicyRunner(OnPolicyRunner):
    """
    OnPolicyRunner extended with constraint (cost) support.
    Collects costs during rollouts and adds constraint penalty to policy loss.
    """
    
    def __init__(
        self,
        env,
        train_cfg,
        log_dir,
        device,
        cost_cfg: Optional[Dict] = None,
    ):
        # Initialize parent
        super().__init__(env, train_cfg, log_dir, device)
        
        # Cost configuration
        self.cost_cfg = cost_cfg
        if cost_cfg is None:
            # Default: no costs
            self.num_costs = 0
            self.cost_names = []
            self.cost_types = {}
            self.cost_thresholds = {}
        else:
            self.cost_names = list(cost_cfg.get("cost_functions", {}).keys())
            self.num_costs = len(self.cost_names)
            self.cost_types = cost_cfg.get("cost_types", {})
            self.cost_thresholds = cost_cfg.get("cost_thresholds", {})
        
        # Initialize constraint algorithm if costs are enabled
        if self.num_costs > 0:
            # Get cost algorithm config from train_cfg (not from alg_cfg to avoid PPO errors)
            cost_alg_cfg = train_cfg.get("cost_algorithm", {})
            # Note: For now, use standard ActorCritic and collect costs for logging
            # Full ConstrainedActorCritic implementation requires custom PPO update
            # self._init_constrained_policy()  # Disabled for now - use standard PPO
            self.constrained_ppo = ConstrainedPPOAlgorithm(
                num_costs=self.num_costs,
                cost_thresholds=self.cost_thresholds,
                cost_names=self.cost_names,
                cost_types=self.cost_types,
                penalty_coef=cost_alg_cfg.get("penalty_coef", 1.0),
                adaptive_threshold=cost_alg_cfg.get("adaptive_threshold", True),
                gamma=self.alg_cfg.get("gamma", 0.99),
                lam=self.alg_cfg.get("lam", 0.95),
                device=str(device),
            )
        
        # Storage for costs
        self.cost_storage = None
        self.cost_value_storage = None
    
    def _init_constrained_policy(self):
        """Initialize ConstrainedActorCritic policy."""
        # Get original policy parameters
        actor_obs_dim = self.alg.num_actor_obs
        critic_obs_dim = self.alg.num_critic_obs
        action_dim = self.alg.num_actions
        
        # Create new constrained actor-critic
        policy_cfg = self.cfg["policy"].copy()
        # Get cost learning rate from train_cfg (not from alg_cfg to avoid PPO errors)
        cost_alg_cfg = self.cfg.get("cost_algorithm", {})
        cost_critic_hidden_dims = policy_cfg.get("cost_critic_hidden_dims", policy_cfg.get("critic_hidden_dims", [512, 256, 128]))
        cost_learning_rate = cost_alg_cfg.get("cost_learning_rate", self.alg_cfg.get("learning_rate", 0.001))
        
        # Remove cost-related params from policy_cfg before passing to ActorCritic
        policy_cfg.pop("cost_critic_hidden_dims", None)
        
        constrained_policy = ConstrainedActorCritic(
            num_actor_obs=actor_obs_dim,
            num_critic_obs=critic_obs_dim,
            num_actions=action_dim,
            num_costs=self.num_costs,
            cost_critic_hidden_dims=cost_critic_hidden_dims,
            cost_learning_rate=cost_learning_rate,
            **policy_cfg,
        ).to(self.device)
        
        # Replace policy
        self.alg.actor_critic = constrained_policy
        self.alg.actor_critic.to(self.device)
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Main learning loop with constraint handling.
        
        Currently uses standard PPO with cost logging.
        Full constraint integration requires modifying rsl-rl-lib's PPO class.
        """
        # For now, use standard OnPolicyRunner but add cost logging
        # This allows training to proceed while collecting cost information
        if self.num_costs == 0:
            # No constraints, use standard training
            super().learn(num_learning_iterations, init_at_random_ep_len)
            return
        
        # Initialize cost storage if needed (for future constraint integration)
        if self.cost_storage is None and self.num_costs > 0:
            num_steps = self.cfg.get("num_steps_per_env", 24)
            num_envs = self.env.num_envs
            self.cost_storage = torch.zeros((num_steps, num_envs, self.num_costs), device=self.device)
            self.cost_value_storage = torch.zeros((num_steps, num_envs, self.num_costs), device=self.device)
        
        # Use standard OnPolicyRunner's learn method but intercept to collect costs
        # This is a simplified approach - full constraint integration requires
        # implementing custom PPO update with constraint penalty
        print(f"\n=== Constrained PPO Training (Cost Logging Mode) ===")
        print(f"Cost constraints: {self.cost_thresholds}")
        print(f"Note: Using standard PPO with cost logging.")
        print(f"Full constraint penalty integration requires custom PPO implementation.\n")
        
        # Call parent learn - costs will be logged via extras['cost'] from environment
        super().learn(num_learning_iterations, init_at_random_ep_len)

