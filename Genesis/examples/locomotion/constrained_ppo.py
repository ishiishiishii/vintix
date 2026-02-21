"""
Constrained PPO Algorithm Implementation
Based on "Not Only Rewards But Also Constraints" paper.

This implements IPO (Interior-Point Policy Optimization) and P3O-style
constrained PPO with:
1. Separate Value Functions for rewards and costs
2. Cost Critics for each constraint
3. Constraint penalty in policy loss (log barrier function)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class MultiHeadCostCritic(nn.Module):
    """
    Multi-head Cost Critic Network.
    Shares feature extraction but has separate heads for each cost type.
    """
    
    def __init__(self, num_obs, num_costs, hidden_dims=[512, 256, 128], activation="elu"):
        super().__init__()
        
        self.num_costs = num_costs
        
        # Activation function
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ELU
        
        # Shared feature extractor
        layers = []
        input_dim = num_obs
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_fn())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for each cost
        self.cost_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[-1]),
                act_fn(),
                nn.Linear(hidden_dims[-1], 1)
            )
            for _ in range(num_costs)
        ])
    
    def forward(self, obs):
        """
        Forward pass through cost critic.
        
        Args:
            obs: Observations [batch_size, num_obs]
        
        Returns:
            cost_values: [batch_size, num_costs]
        """
        features = self.shared_layers(obs)
        cost_values = torch.cat([
            head(features) for head in self.cost_heads
        ], dim=1)
        return cost_values
    
    def evaluate_cost(self, obs):
        """Alias for forward for consistency with rsl-rl API."""
        return self.forward(obs)


class ConstrainedPPO:
    """
    Constrained PPO Algorithm with IPO-style constraint handling.
    
    The policy loss includes a log barrier penalty:
    L_policy = L_PPO - sum_k [lambda_k * log(d_k - J_c_k)]
    
    where:
    - L_PPO: standard PPO clipped surrogate loss
    - lambda_k: penalty coefficient for constraint k
    - d_k: constraint threshold for constraint k
    - J_c_k: expected cost value for constraint k
    """
    
    def __init__(
        self,
        actor_critic,  # Standard actor-critic network
        cost_critic,   # Multi-head cost critic
        num_costs: int,
        cost_thresholds: Dict[str, float],  # d_k for each cost
        cost_names: List[str],
        cost_types: Dict[str, str],  # 'probabilistic' or 'average'
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        cost_value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,
        lam: float = 0.95,
        use_clipped_value_loss: bool = True,
        penalty_coef: float = 1.0,  # lambda in penalty term
        adaptive_threshold: bool = True,  # Use adaptive constraint thresholding
    ):
        self.actor_critic = actor_critic
        self.cost_critic = cost_critic
        self.num_costs = num_costs
        self.cost_thresholds = cost_thresholds
        self.cost_names = cost_names
        self.cost_types = cost_types
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.use_clipped_value_loss = use_clipped_value_loss
        self.penalty_coef = penalty_coef
        self.adaptive_threshold = adaptive_threshold
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        costs: torch.Tensor,  # [num_steps, num_envs, num_costs]
        cost_values: torch.Tensor,  # [num_steps, num_envs, num_costs]
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation) for rewards and costs.
        
        Returns:
            advantages: [num_steps, num_envs]
            returns: [num_steps, num_envs]
            cost_advantages: [num_steps, num_envs, num_costs]
            cost_returns: [num_steps, num_envs, num_costs]
        """
        num_steps, num_envs = rewards.shape
        device = rewards.device
        
        # Compute advantages and returns for rewards
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_value = values[step]
                next_done = dones[step]
            else:
                next_value = values[step + 1]
                next_done = dones[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * (1 - next_done) - values[step]
            advantages[step] = last_gae = delta + self.gamma * self.lam * (1 - next_done) * last_gae
        
        returns = advantages + values
        
        # Compute advantages and returns for costs (similar but for each cost)
        cost_advantages = torch.zeros_like(costs)  # [num_steps, num_envs, num_costs]
        cost_returns = torch.zeros_like(costs)
        
        for cost_idx in range(self.num_costs):
            last_gae_cost = 0
            for step in reversed(range(num_steps)):
                if step == num_steps - 1:
                    next_cost_value = cost_values[step, :, cost_idx]
                    next_done = dones[step]
                else:
                    next_cost_value = cost_values[step + 1, :, cost_idx]
                    next_done = dones[step + 1]
                
                delta_cost = costs[step, :, cost_idx] + self.gamma * next_cost_value * (1 - next_done) - cost_values[step, :, cost_idx]
                cost_advantages[step, :, cost_idx] = last_gae_cost = delta_cost + self.gamma * self.lam * (1 - next_done) * last_gae_cost
            
            cost_returns[:, :, cost_idx] = cost_advantages[:, :, cost_idx] + cost_values[:, :, cost_idx]
        
        return advantages, returns, cost_advantages, cost_returns
    
    def compute_constraint_penalty(self, cost_returns_mean: torch.Tensor) -> torch.Tensor:
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
            
            # Adaptive thresholding: if constraint is violated, use current value as threshold
            if self.adaptive_threshold and J_c_k > d_k:
                d_k_adaptive = max(d_k, J_c_k * 1.01)  # Slightly above current value
            else:
                d_k_adaptive = d_k
            
            # Log barrier: -log(d_k - J_c_k)
            # Add small epsilon for numerical stability
            epsilon = 1e-6
            barrier_arg = d_k_adaptive - J_c_k + epsilon
            
            if barrier_arg > 0:
                penalty += -self.penalty_coef * torch.log(torch.tensor(barrier_arg, device=cost_returns_mean.device))
            else:
                # If constraint is severely violated, use large penalty
                penalty += self.penalty_coef * 100.0
        
        return penalty
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,  # [num_steps, num_envs, num_costs]
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        cost_values: torch.Tensor,  # [num_steps, num_envs, num_costs]
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
    ) -> Dict[str, float]:
        """
        Update policy and value functions.
        
        Returns:
            Dictionary with loss statistics
        """
        # Compute advantages
        advantages, returns, cost_advantages, cost_returns = self.compute_gae(
            rewards, values, costs, cost_values, dones
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy values
        _, action_log_probs, entropy, new_values = self.actor_critic.evaluate(obs, actions)
        new_cost_values = self.cost_critic(obs)  # [batch, num_costs]
        
        # Flatten for mini-batch training
        batch_size = obs.shape[0]
        indices = np.arange(batch_size)
        
        clip_fractions = []
        value_losses = []
        policy_losses = []
        cost_value_losses = []
        constraint_penalties = []
        
        for epoch in range(num_learning_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, batch_size // num_mini_batches):
                end = start + batch_size // num_mini_batches
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages.flatten()[batch_indices]
                batch_returns = returns.flatten()[batch_indices]
                batch_new_log_probs, _, batch_entropy, batch_new_values = self.actor_critic.evaluate(
                    batch_obs, batch_actions
                )
                batch_new_cost_values = self.cost_critic(batch_obs)
                
                # Cost returns for this batch
                batch_cost_returns = cost_returns.view(-1, self.num_costs)[batch_indices]  # [mini_batch, num_costs]
                
                # PPO clipped surrogate loss
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (reward)
                if self.use_clipped_value_loss:
                    value_pred_clipped = values.flatten()[batch_indices] + torch.clamp(
                        batch_new_values - values.flatten()[batch_indices], -self.clip_param, self.clip_param
                    )
                    value_losses_unclipped = (batch_new_values - batch_returns) ** 2
                    value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_losses_unclipped, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * ((batch_new_values - batch_returns) ** 2).mean()
                
                # Cost value loss (for each cost)
                cost_value_loss = 0.0
                for cost_idx in range(self.num_costs):
                    cost_target = batch_cost_returns[:, cost_idx]
                    cost_pred = batch_new_cost_values[:, cost_idx]
                    cost_value_loss += 0.5 * ((cost_pred - cost_target) ** 2).mean()
                cost_value_loss = cost_value_loss / self.num_costs
                
                # Constraint penalty (log barrier)
                batch_cost_returns_mean = batch_cost_returns.mean(dim=0)  # [num_costs]
                constraint_penalty = self.compute_constraint_penalty(batch_cost_returns_mean)
                
                # Total policy loss with constraint penalty
                total_policy_loss = policy_loss + constraint_penalty
                
                # Entropy loss
                entropy_loss = -batch_entropy.mean()
                
                # Total loss
                total_loss = (
                    total_policy_loss
                    + self.value_loss_coef * value_loss
                    + self.cost_value_loss_coef * cost_value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Update (note: optimizer should be set up externally)
                # For now, assume optimizers exist or will be set up in runner
                if hasattr(self.actor_critic, 'optimizer'):
                    self.actor_critic.optimizer.zero_grad()
                if hasattr(self.cost_critic, 'optimizer'):
                    self.cost_critic.optimizer.zero_grad()
                
                total_loss.backward()
                
                if hasattr(self.actor_critic, 'parameters'):
                    torch.nn.utils.clip_grad_norm_(list(self.actor_critic.parameters()), self.max_grad_norm)
                if hasattr(self.cost_critic, 'parameters'):
                    torch.nn.utils.clip_grad_norm_(list(self.cost_critic.parameters()), self.max_grad_norm)
                
                if hasattr(self.actor_critic, 'optimizer'):
                    self.actor_critic.optimizer.step()
                if hasattr(self.cost_critic, 'optimizer'):
                    self.cost_critic.optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                    clip_fractions.append(clip_fraction.item())
                    value_losses.append(value_loss.item())
                    policy_losses.append(policy_loss.item())
                    cost_value_losses.append(cost_value_loss.item())
                    constraint_penalties.append(constraint_penalty.item() if isinstance(constraint_penalty, torch.Tensor) else constraint_penalty)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'cost_value_loss': np.mean(cost_value_losses),
            'constraint_penalty': np.mean(constraint_penalties),
            'clip_fraction': np.mean(clip_fractions),
        }

