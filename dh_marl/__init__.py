"""DH-MARL: Decentralized Hemodynamic-Aware Multi-Agent Reinforcement Learning.

Reference implementation for Huynh & Nguyen, Electronics 2026.
"""

__version__ = "0.1.0"

from dh_marl.models.policy import PerAgentPolicy
from dh_marl.models.critic import GraphAttentionCritic
from dh_marl.models.message_encoder import MessageEncoder
from dh_marl.env.vascular_env import VascularSwarmEnv
from dh_marl.comm.diffusion_channel import DiffusionChannel

__all__ = [
    "PerAgentPolicy",
    "GraphAttentionCritic",
    "MessageEncoder",
    "VascularSwarmEnv",
    "DiffusionChannel",
]
