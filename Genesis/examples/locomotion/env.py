"""
Environment classes for different robots.
This module imports all environment classes from separate files.
"""

from go2_env import Go2Env
from minicheetah_env import MiniCheetahEnv
from laikago_env import LaikagoEnv
from a1_env import A1Env
from anymalc_env import ANYmalCEnv
from go1_env import Go1Env

__all__ = [
    "Go2Env",
    "MiniCheetahEnv",
    "LaikagoEnv",
    "A1Env",
    "ANYmalCEnv",
    "Go1Env",
]
