"""
OmniSafe Environment Wrapper for Genesis Locomotion Environments

This module wraps Genesis locomotion environments to be compatible with OmniSafe.
OmniSafe expects Gym/Gymnasium-style environments with specific methods.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces

import genesis as gs


class GenesisOmniSafeWrapper(gym.Env):
    """
    Wrapper to make Genesis locomotion environments compatible with OmniSafe.
    
    OmniSafe expects:
    - Gym/Gymnasium-style environment interface
    - observation_space and action_space
    - reset() and step() methods
    - Cost information in info dict
    """
    
    def __init__(
        self,
        genesis_env,  # Genesis environment (e.g., ConstrainedGo2Env)
        device: str = "cuda",
    ):
        self.env = genesis_env
        self.device = device
        self.num_envs = genesis_env.num_envs
        
        # Observation space (single environment)
        obs_dim = genesis_env.num_obs
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space (single environment)
        action_dim = genesis_env.num_actions
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        
        # Cost dimension
        if hasattr(genesis_env, 'cost_buf'):
            self.num_costs = genesis_env.cost_buf.shape[1]
        else:
            self.num_costs = 0
        
        # Current observation and cost
        self.current_obs = None
        self.current_cost = None
        
        # For vectorized environments (OmniSafe may use this)
        self.is_vectorized = True
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and return initial observation.
        
        Returns:
            observation: Initial observation
            info: Info dictionary
        """
        obs, info = self.env.reset()
        
        # Convert to numpy if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        # Handle vectorized case (use first env if multiple)
        if len(obs.shape) > 1:
            obs = obs[0]  # Use first environment for single-env interface
        
        self.current_obs = obs.astype(np.float32)
        
        # Extract cost if available
        if hasattr(self.env, 'cost_buf') and self.env.cost_buf is not None:
            cost = self.env.cost_buf[0].cpu().numpy() if isinstance(self.env.cost_buf, torch.Tensor) else self.env.cost_buf[0]
        else:
            cost = np.zeros(self.num_costs, dtype=np.float32)
        
        self.current_cost = cost
        
        info_dict = {
            "cost": cost,
            "cost_names": getattr(self.env, 'cost_names', []),
        }
        
        return obs.astype(np.float32), info_dict
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward scalar
            terminated: Whether episode terminated
            truncated: Whether episode truncated (time limit)
            info: Info dictionary with cost information
        """
        # Convert action to tensor if needed
        if isinstance(action, np.ndarray):
            # Ensure correct shape for vectorized env
            if len(action.shape) == 1:
                action = action.reshape(1, -1)  # Add batch dimension
            action_tensor = torch.from_numpy(action).float().to(self.device)
        else:
            action_tensor = action
        
        # Step environment
        obs, reward, done, info = self.env.step(action_tensor)
        
        # Convert to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()
        
        # Handle vectorized case (use first env if multiple)
        if len(obs.shape) > 1:
            obs = obs[0]
            reward = reward[0].item() if isinstance(reward, np.ndarray) else reward[0]
            done = done[0].item() if isinstance(done, np.ndarray) else done[0]
        
        # Extract cost from info
        if 'cost' in info:
            cost = info['cost']
            if isinstance(cost, torch.Tensor):
                cost = cost[0].cpu().numpy() if len(cost.shape) > 1 else cost.cpu().numpy()
            elif isinstance(cost, np.ndarray) and len(cost.shape) > 1:
                cost = cost[0]
        else:
            cost = np.zeros(self.num_costs, dtype=np.float32)
        
        self.current_obs = obs.astype(np.float32)
        self.current_cost = cost
        
        # OmniSafe expects cost in info dict
        info_dict = {
            "cost": cost,
            "cost_names": getattr(self.env, 'cost_names', []),
            **{k: v for k, v in info.items() if k not in ['cost', 'cost_names']}
        }
        
        # Gymnasium API: terminated and truncated
        terminated = bool(done)
        truncated = False  # Time limit handling can be added here
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info_dict
    
    def render(self):
        """Render environment (if supported)."""
        # Genesis environments handle rendering internally
        pass
    
    def close(self):
        """Close environment."""
        # Cleanup if needed
        pass


class VectorizedGenesisOmniSafeWrapper:
    """
    Vectorized wrapper for OmniSafe that properly handles multiple environments.
    
    OmniSafe's algorithms typically work with vectorized environments.
    This wrapper maintains the vectorized structure while providing
    OmniSafe-compatible interface.
    """
    
    def __init__(
        self,
        genesis_env,  # Genesis environment (already vectorized)
        device: str = "cuda",
    ):
        self.env = genesis_env
        self.device = device
        self.num_envs = genesis_env.num_envs
        
        # Observation and action spaces (for single env)
        obs_dim = genesis_env.num_obs
        action_dim = genesis_env.num_actions
        
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        
        # Vectorized spaces
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        
        # Cost dimension
        if hasattr(genesis_env, 'cost_buf'):
            self.num_costs = genesis_env.cost_buf.shape[1]
        else:
            self.num_costs = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset all environments."""
        obs, info = self.env.reset()
        
        # Convert to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        # Extract costs
        if hasattr(self.env, 'cost_buf') and self.env.cost_buf is not None:
            costs = self.env.cost_buf.cpu().numpy() if isinstance(self.env.cost_buf, torch.Tensor) else self.env.cost_buf
        else:
            costs = np.zeros((self.num_envs, self.num_costs), dtype=np.float32)
        
        info_dict = {
            "cost": costs,
            "cost_names": getattr(self.env, 'cost_names', []),
        }
        
        return obs.astype(np.float32), info_dict
    
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Step all environments.
        
        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        # Convert actions to tensor
        if isinstance(actions, np.ndarray):
            actions_tensor = torch.from_numpy(actions).float().to(self.device)
        else:
            actions_tensor = actions
        
        # Step environment
        obs, rewards, dones, info = self.env.step(actions_tensor)
        
        # Convert to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
        
        # Extract costs
        if 'cost' in info:
            costs = info['cost']
            if isinstance(costs, torch.Tensor):
                costs = costs.cpu().numpy()
        else:
            costs = np.zeros((self.num_envs, self.num_costs), dtype=np.float32)
        
        # Convert done to terminated and truncated
        terminateds = dones.astype(bool)
        truncateds = np.zeros_like(terminateds, dtype=bool)
        
        info_dict = {
            "cost": costs,
            "cost_names": getattr(self.env, 'cost_names', []),
            **{k: v for k, v in info.items() if k not in ['cost', 'cost_names']}
        }
        
        return (
            obs.astype(np.float32),
            rewards.astype(np.float32),
            terminateds,
            truncateds,
            info_dict,
        )


