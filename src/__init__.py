"""
Snake Reinforcement Learning Package

A modern implementation of Q-Learning applied to the classic Snake game.
"""

__version__ = '2.0.0'
__author__ = 'Kacper Kowalski'

from .Snake import SnakeGame, Snake, BodyNode
from .QLearningAgent import QLearningAgent
from .config import (
    GameConfig,
    TrainingConfig,
    VisualizationConfig,
    PathConfig,
    PresetConfigs
)

__all__ = [
    'SnakeGame',
    'Snake',
    'BodyNode',
    'QLearningAgent',
    'GameConfig',
    'TrainingConfig',
    'VisualizationConfig',
    'PathConfig',
    'PresetConfigs',
]
